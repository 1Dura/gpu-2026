#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#define BlockRows 128
#define BlockCols 128
#define StepK 8
#define ThreadRows 8
#define ThreadCols 8
#define ThreadSize 16
#define Lines 8

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n);

#endif // __BLOCK_GEMM_CUDA_H