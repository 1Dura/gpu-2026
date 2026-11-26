#define CL_TARGET_OPENCL_VERSION 120
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <future>
#include <stdexcept>
#include <string>
#ifdef __GLIBC__
#include <malloc.h>
#endif

static const char *kSource = R"CLC(
__kernel void gelu(__global const float *in, __global float *out, ulong n) {
    const float k = -1.5957691216057308f;
    ulong i = get_global_id(0) * 4;
    if (i + 3 < n) {
        float4 x = vload4(0, in + i);
        float4 t = fmin(k * (x + 0.044715f * x * x * x), 88.0f);
        vstore4(x / (1.0f + exp(t)), 0, out + i);
    } else {
        for (; i < n; ++i) {
            float x = in[i];
            float t = fmin(k * (x + 0.044715f * x * x * x), 88.0f);
            out[i] = x / (1.0f + exp(t));
        }
    }
}
)CLC";

static void Check(cl_int err, const char *what) {
  if (err != CL_SUCCESS)
    throw std::runtime_error(std::string("GeluOCL: ") + what +
                             " failed, OpenCL error " + std::to_string(err));
}

struct OclState {
  int platform = -1;
  cl_context ctx = nullptr;
  cl_command_queue q = nullptr;
  cl_program prog = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem d_in = nullptr, d_out = nullptr;
  size_t allocated = 0;

  void Release() {
    if (d_in) clReleaseMemObject(d_in);
    if (d_out) clReleaseMemObject(d_out);
    if (kernel) clReleaseKernel(kernel);
    if (prog) clReleaseProgram(prog);
    if (q) clReleaseCommandQueue(q);
    if (ctx) clReleaseContext(ctx);
    d_in = d_out = nullptr;
    kernel = nullptr;
    prog = nullptr;
    q = nullptr;
    ctx = nullptr;
    allocated = 0;
    platform = -1;
  }

  void Init(int p) {
    Release();
    cl_uint np = 0;
    Check(clGetPlatformIDs(0, nullptr, &np), "clGetPlatformIDs");
    if (p < 0 || (cl_uint)p >= np)
      throw std::runtime_error("GeluOCL: invalid platform index " + std::to_string(p));
    std::vector<cl_platform_id> plats(np);
    Check(clGetPlatformIDs(np, plats.data(), nullptr), "clGetPlatformIDs");
    cl_device_id dev;
    Check(clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr), "clGetDeviceIDs");

    cl_int err;
    ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    q = clCreateCommandQueue(ctx, dev, 0, &err);
    Check(err, "clCreateCommandQueue");
    prog = clCreateProgramWithSource(ctx, 1, &kSource, nullptr, &err);
    Check(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t len = 0;
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
      std::string log(len, '\0');
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
      throw std::runtime_error("GeluOCL: kernel build failed:\n" + log);
    }
    kernel = clCreateKernel(prog, "gelu", &err);
    Check(err, "clCreateKernel");
    platform = p;
  }

  void Reserve(size_t bytes) {
    if (bytes <= allocated)
      return;
    if (d_in) clReleaseMemObject(d_in);
    if (d_out) clReleaseMemObject(d_out);
    d_in = d_out = nullptr;
    allocated = 0;
    cl_int err;
    d_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    Check(err, "clCreateBuffer(in)");
    d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    Check(err, "clCreateBuffer(out)");
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
  static bool malloc_tuned = false;
  if (!malloc_tuned) {
    TuneMalloc();
    malloc_tuned = true;
  }
  if (st.platform != platform)
    st.Init(platform);

  size_t n = input.size();
  if (n == 0)
    return {};
  size_t bytes = n * sizeof(float);

  auto fut = std::async(std::launch::async, [n] { return std::vector<float>(n); });

  st.Reserve(bytes);
  Check(clEnqueueWriteBuffer(st.q, st.d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr,
                             nullptr),
        "clEnqueueWriteBuffer");
  cl_ulong cn = n;
  Check(clSetKernelArg(st.kernel, 0, sizeof(cl_mem), &st.d_in), "clSetKernelArg(0)");
  Check(clSetKernelArg(st.kernel, 1, sizeof(cl_mem), &st.d_out), "clSetKernelArg(1)");
  Check(clSetKernelArg(st.kernel, 2, sizeof(cl_ulong), &cn), "clSetKernelArg(2)");
  size_t local = 256;
  size_t global = ((n + 3) / 4 + local - 1) / local * local;
  Check(clEnqueueNDRangeKernel(st.q, st.kernel, 1, nullptr, &global, &local, 0, nullptr,
                               nullptr),
        "clEnqueueNDRangeKernel");
  Check(clFlush(st.q), "clFlush");

  std::vector<float> res = fut.get();
  Check(clEnqueueReadBuffer(st.q, st.d_out, CL_TRUE, 0, bytes, res.data(), 0, nullptr,
                            nullptr),
        "clEnqueueReadBuffer");
  return res;
}
