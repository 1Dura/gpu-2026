#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <vector>

static const char* kernel_source = R"(
__kernel void gelu_kernel(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float x = in[i];
        float arg = -1.59576912f * (x + 0.044715f * x * x * x);
        out[i] = x / (1.0f + exp2(arg));
    }
}
)";

struct OCLState {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity_bytes = 0;
    int current_platform = -1;
};

static OCLState state;

static void init_opencl(int platform_idx) {
    if (state.context && state.current_platform == platform_idx) {
        return;
    }

    if (state.kernel) clReleaseKernel(state.kernel);
    if (state.queue) clReleaseCommandQueue(state.queue);
    if (state.context) clReleaseContext(state.context);
    if (state.d_in) clReleaseMemObject(state.d_in);
    if (state.d_out) clReleaseMemObject(state.d_out);
    state = OCLState();

    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_device_id device;
    clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    state.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    state.queue = clCreateCommandQueue(state.context, device, 0, nullptr);

    cl_program program = clCreateProgramWithSource(state.context, 1, &kernel_source, nullptr, nullptr);

    clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);

    state.kernel = clCreateKernel(program, "gelu_kernel", nullptr);
    clReleaseProgram(program);
    
    state.current_platform = platform_idx;
}

static void ensure_capacity(size_t bytes) {
    if (state.capacity_bytes >= bytes) {
        return;
    }
    if (state.d_in) clReleaseMemObject(state.d_in);
    if (state.d_out) clReleaseMemObject(state.d_out);
    
    state.d_in = clCreateBuffer(state.context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
    state.d_out = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
    state.capacity_bytes = bytes;
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    size_t n = input.size();
    if (n == 0) return {};

    size_t bytes = n * sizeof(float);

    init_opencl(platform);
    ensure_capacity(bytes);

    clEnqueueWriteBuffer(state.queue, state.d_in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    unsigned int n_arg = static_cast<unsigned int>(n);
    clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &state.d_in);
    clSetKernelArg(state.kernel, 1, sizeof(cl_mem), &state.d_out);
    clSetKernelArg(state.kernel, 2, sizeof(unsigned int), &n_arg);

    size_t local_size = 256;
    size_t global_size = ((n + local_size - 1) / local_size) * local_size;

    clEnqueueNDRangeKernel(state.queue, state.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);

    std::vector<float> result(n);

    clEnqueueReadBuffer(state.queue, state.d_out, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr);

    return result;
}