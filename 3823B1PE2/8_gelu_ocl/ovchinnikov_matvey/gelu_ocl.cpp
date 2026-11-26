#define CL_TARGET_OPENCL_VERSION 120
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <algorithm>
#include <future>
#ifdef __GLIBC__
#include <malloc.h>
#endif

static const char *kSource = R"CLC(
__kernel void gelu(__global const float *in, __global float *out, ulong off, ulong n) {
    const float k = -1.5957691216057308f; // -2 * sqrt(2 / pi)
    ulong i = off + get_global_id(0) * 4;
    if (i + 3 < n) {
        float4 x = vload4(0, in + i);
        float4 t = k * (x + 0.044715f * x * x * x);
        vstore4(x / (1.0f + native_exp(t)), 0, out + i);
    } else {
        for (; i < n; ++i) {
            float x = in[i];
            out[i] = x / (1.0f + native_exp(k * (x + 0.044715f * x * x * x)));
        }
    }
}
)CLC";

#define LINES 8

struct OclState {
  int platform = -1;
  cl_context ctx = nullptr;
  cl_command_queue q[LINES] = {};
  cl_kernel kernel = nullptr;
  cl_mem d_in = nullptr, d_out = nullptr;
  size_t allocated = 0;

  void Init(int p) {
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    cl_device_id dev;
    clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, nullptr);
    for (int i = 0; i < LINES; ++i)
      q[i] = clCreateCommandQueue(ctx, dev, 0, nullptr);
    cl_program prog =
        clCreateProgramWithSource(ctx, 1, &kSource, nullptr, nullptr);
    clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr);
    kernel = clCreateKernel(prog, "gelu", nullptr);
    clReleaseProgram(prog);
    platform = p;
  }

  void Reserve(size_t bytes) {
    if (bytes <= allocated)
      return;
    if (d_in)
      clReleaseMemObject(d_in);
    if (d_out)
      clReleaseMemObject(d_out);
    d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
    allocated = bytes;
  }
};

static void TuneMalloc() {
#ifdef __GLIBC__
  mallopt(M_MMAP_MAX, 0);
  mallopt(M_TRIM_THRESHOLD, -1);
  mallopt(M_ARENA_MAX, 1);
#endif
}

std::vector<float> GeluOCL(const std::vector<float> &input, int platform) {
  static OclState st;
  if (st.platform != platform) {
    TuneMalloc();
    st.Init(platform);
  }

  size_t n = input.size();
  auto fut =
      std::async(std::launch::async, [n] { return std::vector<float>(n); });
  if (n == 0)
    return fut.get();
  st.Reserve(n * sizeof(float));

  size_t lines = n >= LINES * 4096 ? LINES : 1;
  size_t chunk = ((n + lines - 1) / lines + 3) / 4 * 4; // округление ВВЕРХ
  for (size_t i = 0; i < lines; ++i) {
    size_t off = i * chunk;
    if (off >= n) {
      lines = i;
      break;
    }
    size_t cnt = std::min(chunk, n - off);
    clEnqueueWriteBuffer(st.q[i], st.d_in, CL_FALSE, off * 4, cnt * 4,
                         input.data() + off, 0, nullptr, nullptr);
    cl_ulong a_off = off, a_end = off + cnt;
    clSetKernelArg(st.kernel, 0, sizeof(cl_mem), &st.d_in);
    clSetKernelArg(st.kernel, 1, sizeof(cl_mem), &st.d_out);
    clSetKernelArg(st.kernel, 2, sizeof(cl_ulong), &a_off);
    clSetKernelArg(st.kernel, 3, sizeof(cl_ulong), &a_end);
    size_t local = 256;
    size_t global = ((cnt + 3) / 4 + local - 1) / local * local;
    clEnqueueNDRangeKernel(st.q[i], st.kernel, 1, nullptr, &global, &local, 0,
                           nullptr, nullptr);
  }

  std::vector<float> res = fut.get();
  for (size_t i = 0; i < lines; ++i) {
    size_t off = i * chunk;
    size_t cnt = std::min(chunk, n - off);
    clEnqueueReadBuffer(st.q[i], st.d_out, CL_FALSE, off * 4, cnt * 4,
                        res.data() + off, 0, nullptr, nullptr);
  }
  for (size_t i = 0; i < lines; ++i)
    clFinish(st.q[i]);
  return res;
}
