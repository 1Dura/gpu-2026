#include "softmax_cuda.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <future>
#include <malloc.h>
#include <thread>

#define THREADS 256

__device__ float WarpMax(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}

__device__ float WarpSum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

__device__ float BlockMax(float v) {
  __shared__ float buf[32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  int warps = blockDim.x / 32;

  v = WarpMax(v);
  if (lane == 0)
    buf[warp] = v;
  __syncthreads();

  if (warp == 0) {
    v = (lane < warps) ? buf[lane] : -INFINITY;
    v = WarpMax(v);
    if (lane == 0)
      buf[0] = v;
  }
  __syncthreads();
  float res = buf[0];
  __syncthreads();
  return res;
}

__device__ float BlockSum(float v) {
  __shared__ float buf[32];
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  int warps = blockDim.x / 32;

  v = WarpSum(v);
  if (lane == 0)
    buf[warp] = v;
  __syncthreads();

  if (warp == 0) {
    v = (lane < warps) ? buf[lane] : 0.0f;
    v = WarpSum(v);
    if (lane == 0)
      buf[0] = v;
  }
  __syncthreads();
  float res = buf[0];
  __syncthreads();
  return res;
}
__global__ void SoftmaxKernel(const float *in, float *out, int cols) {
  const float *x = in + (size_t)blockIdx.x * cols;
  float *y = out + (size_t)blockIdx.x * cols;

  float m = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    m = fmaxf(m, x[i]);
  m = BlockMax(m);

  float s = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    s += __expf(x[i] - m);
  s = BlockSum(s);

  float inv = 1.0f / s;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    y[i] = __expf(x[i] - m) * inv;
}
template <int VEC>
__global__ void SoftmaxKernelReg(const float *in, float *out, int cols) {
  const float4 *x =
      reinterpret_cast<const float4 *>(in + (size_t)blockIdx.x * cols);
  float4 *y = reinterpret_cast<float4 *>(out + (size_t)blockIdx.x * cols);

  float4 v[VEC];
  float m = -INFINITY;
#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    v[k] = x[threadIdx.x + k * blockDim.x];
    m = fmaxf(m, fmaxf(fmaxf(v[k].x, v[k].y), fmaxf(v[k].z, v[k].w)));
  }
  m = BlockMax(m);

  float s = 0.0f;
#pragma unroll
  for (int k = 0; k < VEC; ++k) {
    v[k].x = __expf(v[k].x - m);
    v[k].y = __expf(v[k].y - m);
    v[k].z = __expf(v[k].z - m);
    v[k].w = __expf(v[k].w - m);
    s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
  }
  float inv = 1.0f / BlockSum(s);

#pragma unroll
  for (int k = 0; k < VEC; ++k)
    y[threadIdx.x + k * blockDim.x] =
        make_float4(v[k].x * inv, v[k].y * inv, v[k].z * inv, v[k].w * inv);
}
void LaunchSoftmax(const float *in, float *out, int rows, int cols,
                   cudaStream_t st) {
  if (cols % 16 == 0) {
    int t4 = cols / 16; // VEC = 4 -> 16 чисел на поток
    if (t4 >= 128 && t4 <= 1024 && t4 % 32 == 0) {
      SoftmaxKernelReg<4><<<rows, t4, 0, st>>>(in, out, cols);
      return;
    }
  }
  if (cols % 32 == 0) {
    int t8 = cols / 32; // VEC = 8 -> 32 числа на поток
    if (t8 >= 128 && t8 <= 1024 && t8 % 32 == 0) {
      SoftmaxKernelReg<8><<<rows, t8, 0, st>>>(in, out, cols);
      return;
    }
  }
  SoftmaxKernel<<<rows, THREADS, 0, st>>>(in, out, cols);
}

#define SLOTS 3
#define CHUNK_BYTES (32u << 20)

std::vector<float> SoftmaxCUDA(const std::vector<float> &input, int row_count) {
  size_t n = input.size();
  int cols = (int)(n / row_count);
  size_t bytes = n * sizeof(float);

  static bool malloc_tuned = false;
  if (!malloc_tuned) {
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_ARENA_MAX, 1);
    malloc_tuned = true;
  }

  auto fut =
      std::async(std::launch::async, [n] { return std::vector<float>(n); });

  static float *d_in = nullptr, *d_out = nullptr;
  static size_t allocated = 0;
  if (bytes > allocated) {
    cudaFree(d_in);
    cudaFree(d_out);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    allocated = bytes;
  }

  static float *pin_in[SLOTS], *pin_out[SLOTS];
  static cudaStream_t streams[SLOTS];
  static cudaEvent_t h2d_done[SLOTS], d2h_done[SLOTS];
  static bool ready = false;
  if (!ready) {
    for (int s = 0; s < SLOTS; ++s) {
      cudaMallocHost(&pin_in[s], CHUNK_BYTES);
      cudaMallocHost(&pin_out[s], CHUNK_BYTES);
      cudaStreamCreate(&streams[s]);
      cudaEventCreateWithFlags(&h2d_done[s], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&d2h_done[s], cudaEventDisableTiming);
    }
    ready = true;
  }

  size_t row_bytes = (size_t)cols * sizeof(float);
  if (row_bytes > CHUNK_BYTES) {
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);
    LaunchSoftmax(d_in, d_out, row_count, cols, 0);
    std::vector<float> res = fut.get();
    cudaMemcpy(res.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    return res;
  }
  int rows = (int)(CHUNK_BYTES / row_bytes);
  if (rows > row_count)
    rows = row_count;
  int chunks = (row_count + rows - 1) / rows;

  std::atomic<int> issued{0};
  std::atomic<int> consumed{0};
  std::vector<float> res;

  std::thread back([&] {
    res = fut.get();
    for (int i = 0; i < chunks; ++i) {
      int s = i % SLOTS;
      while (issued.load() <= i)
        std::this_thread::yield();
      cudaEventSynchronize(d2h_done[s]);
      int r = std::min(rows, row_count - i * rows);
      size_t off = (size_t)i * rows * cols;
      memcpy(res.data() + off, pin_out[s], (size_t)r * row_bytes);
      consumed.store(i + 1);
    }
  });

  for (int i = 0; i < chunks; ++i) {
    int s = i % SLOTS;
    int r = std::min(rows, row_count - i * rows);
    size_t off = (size_t)i * rows * cols;
    size_t sz = (size_t)r * row_bytes;

    if (i >= SLOTS)
      cudaEventSynchronize(h2d_done[s]);
    memcpy(pin_in[s], input.data() + off, sz);

    while (consumed.load() < i - SLOTS + 1)
      std::this_thread::yield();

    cudaMemcpyAsync(d_in + off, pin_in[s], sz, cudaMemcpyHostToDevice,
                    streams[s]);
    cudaEventRecord(h2d_done[s], streams[s]);
    LaunchSoftmax(d_in + off, d_out + off, r, cols, streams[s]);
    cudaMemcpyAsync(pin_out[s], d_out + off, sz, cudaMemcpyDeviceToHost,
                    streams[s]);
    cudaEventRecord(d2h_done[s], streams[s]);
    issued.store(i + 1);
  }

  back.join();
  return res;
}
