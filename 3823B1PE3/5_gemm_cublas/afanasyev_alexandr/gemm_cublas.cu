#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    const size_t matrix_size = static_cast<size_t>(n) * static_cast<size_t>(n);
    if (a.size() != matrix_size || b.size() != matrix_size) {
        throw std::invalid_argument("A and B must both have size n * n");
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaError_t cuda_err = cudaMalloc(&d_a, matrix_size * sizeof(float));
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc A failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc(&d_b, matrix_size * sizeof(float));
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        throw std::runtime_error(std::string("cudaMalloc B failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc(&d_c, matrix_size * sizeof(float));
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        throw std::runtime_error(std::string("cudaMalloc C failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMemcpy(d_a, a.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy A failed: ") + cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMemcpy(d_b, b.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy B failed: ") + cudaGetErrorString(cuda_err));
    }

    cublasHandle_t handle = nullptr;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cublasCreate failed");
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // For row-major matrices, cuBLAS expects column-major data, so we pass
    // transposed operands to compute C = A * B in row-major output space.
    status = cublasSgemm(handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_T,
                         n,
                         n,
                         n,
                         &alpha,
                         d_a,
                         n,
                         d_b,
                         n,
                         &beta,
                         d_c,
                         n);

    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error("cublasSgemm failed");
    }

    std::vector<float> c(matrix_size);
    cuda_err = cudaMemcpy(c.data(), d_c, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        cublasDestroy(handle);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw std::runtime_error(std::string("cudaMemcpy C failed: ") + cudaGetErrorString(cuda_err));
    }

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
