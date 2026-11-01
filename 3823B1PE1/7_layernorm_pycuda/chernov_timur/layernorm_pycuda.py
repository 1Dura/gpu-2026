import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

LAYERNORM_KERNEL = r"""
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int row_size,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* in_row = input + (long long)row * row_size;
    float* out_row = output + (long long)row * row_size;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float x = in_row[i];
        local_sum += x;
        local_sum_sq += x * x;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    __shared__ float warp_sum[8];
    __shared__ float warp_sum_sq[8];
    if (tid % 32 == 0) {
        warp_sum[tid / 32] = local_sum;
        warp_sum_sq[tid / 32] = local_sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        float s = (tid < 8) ? warp_sum[tid] : 0.0f;
        float sq = (tid < 8) ? warp_sum_sq[tid] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xffffffff, s, offset);
            sq += __shfl_down_sync(0xffffffff, sq, offset);
        }
        if (tid == 0) { 
            warp_sum[0] = s; 
            warp_sum_sq[0] = sq; 
        }
    }
    __syncthreads();

    float mean = warp_sum[0] / row_size;
    float variance = (warp_sum_sq[0] / row_size) - (mean * mean);
    float inv_std = rsqrtf(variance + eps);

    for (int i = tid; i < row_size; i += blockDim.x) {
        float x = in_row[i];
        out_row[i] = gamma[i] * (x - mean) * inv_std + beta[i];
    }
}
"""

_module = SourceModule(LAYERNORM_KERNEL)
_kernel = _module.get_function("layernorm_kernel")

class _Workspace:
    def __init__(self):
        self.d_in = None
        self.d_out = None
        self.d_gamma = None
        self.d_beta = None
        self.cached_n = 0
        self.cached_nf = 0
        self.stream = cuda.Stream()

    def ensure_capacity(self, n, nf):
        if self.cached_n < n:
            if self.d_in: self.d_in.free()
            if self.d_out: self.d_out.free()
            self.d_in = cuda.mem_alloc(n * 4)
            self.d_out = cuda.mem_alloc(n * 4)
            self.cached_n = n
            
        if self.cached_nf < nf:
            if self.d_gamma: self.d_gamma.free()
            if self.d_beta: self.d_beta.free()
            self.d_gamma = cuda.mem_alloc(nf * 4)
            self.d_beta = cuda.mem_alloc(nf * 4)
            self.cached_nf = nf

_workspace = _Workspace()

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    inp = np.asarray(input, dtype=np.float32).ravel()
    gam = np.asarray(gamma, dtype=np.float32).ravel()
    bet = np.asarray(beta, dtype=np.float32).ravel()

    n = inp.size
    nf = int(row_size)
    nr = n // nf

    _workspace.ensure_capacity(n, nf)

    h_in = cuda.pagelocked_empty(n, dtype=np.float32)
    h_g = cuda.pagelocked_empty(nf, dtype=np.float32)
    h_b = cuda.pagelocked_empty(nf, dtype=np.float32)
    h_out = cuda.pagelocked_empty(n, dtype=np.float32)

    np.copyto(h_in, inp)
    np.copyto(h_g, gam)
    np.copyto(h_b, bet)

    stream = _workspace.stream
    
    cuda.memcpy_htod_async(_workspace.d_in, h_in, stream)
    cuda.memcpy_htod_async(_workspace.d_gamma, h_g, stream)
    cuda.memcpy_htod_async(_workspace.d_beta, h_b, stream)

    threads = min(1024, nf)
    threads = 1 << (threads.bit_length() - 1)

    _kernel(
        _workspace.d_in, _workspace.d_gamma, _workspace.d_beta, _workspace.d_out,
        np.int32(nf), np.float32(eps),
        block=(threads, 1, 1), grid=(nr, 1, 1), stream=stream
    )

    cuda.memcpy_dtoh_async(h_out, _workspace.d_out, stream)
    stream.synchronize()

    return h_out