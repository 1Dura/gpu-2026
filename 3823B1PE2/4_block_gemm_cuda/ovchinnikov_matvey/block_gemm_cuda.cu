#include "block_gemm_cuda.h"
#include <future>

__global__ void KernelS(float *c, const float *a, const float *b, int n) {
  __shared__ float As[ThreadSize][ThreadSize];
  __shared__ float Bs[ThreadSize][ThreadSize];

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * ThreadSize + ty;
  int col = blockIdx.x * ThreadSize + tx;
  float sum = 0;

  for (int t = 0; t < n; t += ThreadSize) {
    As[ty][tx] = a[row * n + (t + tx)];
    Bs[ty][tx] = b[(t + ty) * n + col];
    __syncthreads();

    for (int k = 0; k < ThreadSize; ++k)
      sum += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  c[row * n + col] = sum;
}

__global__ void Kernel(float *c, const float *a, const float *b, int n) {
  __shared__ float As[StepK][BlockRows];
  __shared__ float Bs[StepK][BlockCols];

  int tid = threadIdx.x;
  int tr = tid / (BlockCols / ThreadCols), tc = tid % (BlockCols / ThreadCols);

  const float *A = a + (size_t)blockIdx.y * BlockRows * n;
  const float *B = b + blockIdx.x * BlockCols;

  int a_row = tid / 2, a_col = (tid % 2) * 4;
  int b_row = tid / 32, b_col = (tid % 32) * 4;

  float acc[ThreadRows][ThreadCols] = {0};
  float ra[ThreadRows], rb[ThreadCols];
  for (int t = 0; t < n; t += StepK) {
    float4 av =
        *reinterpret_cast<const float4 *>(&A[(size_t)a_row * n + t + a_col]);
    As[a_col][a_row] = av.x;
    As[a_col + 1][a_row] = av.y;
    As[a_col + 2][a_row] = av.z;
    As[a_col + 3][a_row] = av.w;

    *reinterpret_cast<float4 *>(&Bs[b_row][b_col]) =
        *reinterpret_cast<const float4 *>(&B[(size_t)(t + b_row) * n + b_col]);

    __syncthreads();
#pragma unroll
    for (int k = 0; k < StepK; ++k) {
#pragma unroll
      for (int i = 0; i < ThreadRows; ++i)
        ra[i] = As[k][tr * ThreadRows + i];
#pragma unroll
      for (int j = 0; j < ThreadCols; ++j)
        rb[j] = Bs[k][tc * ThreadCols + j];
#pragma unroll
      for (int i = 0; i < ThreadRows; ++i)
#pragma unroll
        for (int j = 0; j < ThreadCols; ++j)
          acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  float *C = c + (size_t)(blockIdx.y * BlockRows + tr * ThreadRows) * n +
             blockIdx.x * BlockCols + tc * ThreadCols;
#pragma unroll
  for (int i = 0; i < ThreadRows; ++i)
#pragma unroll
    for (int j = 0; j < ThreadCols; j += 4)
      *reinterpret_cast<float4 *>(&C[(size_t)i * n + j]) =
          make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  size_t s = n * n;

  size_t bs = s * sizeof(float);
  auto fut =
      std::async(std::launch::async, [s] { return std::vector<float>(s); });

  static size_t allocated = 0;
  static float *d_a = nullptr;
  static float *d_b = nullptr;
  static float *d_res = nullptr;
  if (bs > allocated) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaMalloc(&d_a, bs);
    cudaMalloc(&d_b, bs);
    cudaMalloc(&d_res, bs);
    allocated = bs;
  }

  static cudaStream_t streams[Lines];
  static bool streams_ready = false;
  if (!streams_ready) {
    for (int i = 0; i < Lines; ++i)
      cudaStreamCreate(&streams[i]);
    streams_ready = true;
  }

  if (n >= Lines * BlockRows) {
    cudaMemcpy(d_b, b.data(), bs, cudaMemcpyHostToDevice);
    int rows = n / Lines;
    size_t stripe_bytes = (size_t)rows * n * sizeof(float);
    for (int i = 0; i < Lines; ++i) {
      size_t off = (size_t)i * rows * n;
      cudaMemcpyAsync(d_a + off, a.data() + off, stripe_bytes,
                      cudaMemcpyHostToDevice, streams[i]);
      Kernel<<<dim3(n / BlockCols, rows / BlockRows), 256, 0, streams[i]>>>(
          d_res + off, d_a + off, d_b, n);
    }
    std::vector<float> res = fut.get();
    for (int i = 0; i < Lines; ++i) {
      size_t off = (size_t)i * rows * n;
      cudaMemcpyAsync(res.data() + off, d_res + off, stripe_bytes,
                      cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();
    return res;
  }

  cudaMemcpy(d_a, a.data(), bs, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), bs, cudaMemcpyHostToDevice);
  if (n >= BlockRows)
    Kernel<<<dim3(n / BlockCols, n / BlockRows), 256>>>(d_res, d_a, d_b, n);
  else
    KernelS<<<dim3((n + ThreadSize - 1) / ThreadSize,
                   (n + ThreadSize - 1) / ThreadSize),
              dim3(ThreadSize, ThreadSize)>>>(d_res, d_a, d_b, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

  std::vector<float> res = fut.get();
  cudaMemcpy(res.data(), d_res, bs, cudaMemcpyDeviceToHost);
  return res;
}
