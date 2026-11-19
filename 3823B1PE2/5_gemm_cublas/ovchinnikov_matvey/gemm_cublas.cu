#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <future>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int n) {
  static cublasHandle_t handle = nullptr;
  if (!handle)
    cublasCreate(&handle);

  size_t s = n * n;
  size_t bs = s * sizeof(float);
  static size_t allocated = 0;
  static float *d_a = nullptr;
  static float *d_b = nullptr;
  static float *d_res = nullptr;
  auto fut =
      std::async(std::launch::async, [s] { return std::vector<float>(s); });
  if (bs > allocated) {
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    cudaMalloc(&d_a, bs);
    cudaMalloc(&d_b, bs);
    cudaMalloc(&d_res, bs);
    allocated = bs;
  }
  //   cudaMemcpy(d_a, a.data(), bs, cudaMemcpyHostToDevice);
  //   cudaMemcpy(d_b, b.data(), bs, cudaMemcpyHostToDevice);
  //   const float CA = 1.0f, CB = 0.0f;
  //   cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
  //   n,
  //                                   &CA, d_b, n, d_a, n, &CB, d_res, n);

  static cudaStream_t streams[Lines];
  static bool streams_ready = false;
  if (!streams_ready) {
    for (int i = 0; i < Lines; ++i) {
      cudaStreamCreate(&streams[i]);
    }
    streams_ready = true;
  }
  const float CA = 1.0f, CB = 0.0f;

  cudaMemcpy(d_b, b.data(), bs, cudaMemcpyHostToDevice);
  int rows = n / Lines;
  size_t stripe_bytes = (size_t)rows * n * sizeof(float);
  for (int i = 0; i < Lines; ++i) {
    size_t off = (size_t)i * rows * n;
    cudaMemcpyAsync(d_a + off, a.data() + off, stripe_bytes,
                    cudaMemcpyHostToDevice, streams[i]);
    cublasSetStream(handle, streams[i]);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, n, &CA, d_b, n,
                d_a + off, n, &CB, d_res + off, n);
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