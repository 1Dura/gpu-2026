import ctypes
import threading

import numpy as np
import pycuda.autoinit 
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_SRC = r"""
__device__ __forceinline__ float WarpSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ float BlockSum(float v) {
    __shared__ float buf[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    v = WarpSum(v);
    if (lane == 0) buf[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < warps) ? buf[lane] : 0.0f;
        v = WarpSum(v);
        if (lane == 0) buf[0] = v;
    }
    __syncthreads();
    float r = buf[0];
    __syncthreads();
    return r;
}

template <int VEC>
__device__ void LayerNormReg(const float* __restrict__ in, float* __restrict__ out,
                             const float* __restrict__ gamma,
                             const float* __restrict__ beta, int cols, float eps) {
    const float4* x = reinterpret_cast<const float4*>(in + (size_t)blockIdx.x * cols);
    float4* y = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * cols);
    const float4* g = reinterpret_cast<const float4*>(gamma);
    const float4* b = reinterpret_cast<const float4*>(beta);

    float4 v[VEC];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
        v[k] = x[threadIdx.x + k * blockDim.x];
        s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
    }
    float mean = BlockSum(s) / cols;

    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
        float a = v[k].x - mean, c = v[k].y - mean, d = v[k].z - mean, e = v[k].w - mean;
        q += (a * a + c * c) + (d * d + e * e);
    }
    float rstd = rsqrtf(BlockSum(q) / cols + eps);

    #pragma unroll
    for (int k = 0; k < VEC; ++k) {
        int i = threadIdx.x + k * blockDim.x;
        float4 gg = g[i], bb = b[i];
        y[i] = make_float4((v[k].x - mean) * rstd * gg.x + bb.x,
                           (v[k].y - mean) * rstd * gg.y + bb.y,
                           (v[k].z - mean) * rstd * gg.z + bb.z,
                           (v[k].w - mean) * rstd * gg.w + bb.w);
    }
}

extern "C" {
__global__ void ln_reg4(const float* in, float* out, const float* g, const float* b,
                        int cols, float eps) { LayerNormReg<4>(in, out, g, b, cols, eps); }
__global__ void ln_reg8(const float* in, float* out, const float* g, const float* b,
                        int cols, float eps) { LayerNormReg<8>(in, out, g, b, cols, eps); }

__global__ void ln_simple(const float* __restrict__ in, float* __restrict__ out,
                          const float* __restrict__ g, const float* __restrict__ b,
                          int cols, float eps) {
    const float* x = in + (size_t)blockIdx.x * cols;
    float* y = out + (size_t)blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += x[i];
    float mean = BlockSum(s) / cols;
    float q = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { float d = x[i] - mean; q += d * d; }
    float rstd = rsqrtf(BlockSum(q) / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] = (x[i] - mean) * rstd * g[i] + b[i];
}
}
"""

_mod = SourceModule(_SRC, no_extern_c=True, options=["-O3"])
_ln_reg4 = _mod.get_function("ln_reg4")
_ln_reg8 = _mod.get_function("ln_reg8")
_ln_simple = _mod.get_function("ln_simple")
for _f in (_ln_reg4, _ln_reg8, _ln_simple):
    _f.prepare("PPPPif")


def _pick_kernel(cols):
    """Ядро и число потоков в блоке для строки длины cols."""
    for kern, per_thread in ((_ln_reg4, 16), (_ln_reg8, 32)):
        t = cols // per_thread
        if cols % per_thread == 0 and 128 <= t <= 1024 and t % 32 == 0:
            return kern, t
    return _ln_simple, max(32, min(256, cols))


def _tune_malloc():
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return
    M_TRIM_THRESHOLD, M_MMAP_MAX, M_ARENA_MAX = -1, -4, -8
    libc.mallopt(M_MMAP_MAX, 0)
    libc.mallopt(M_TRIM_THRESHOLD, -1)
    libc.mallopt(M_ARENA_MAX, 1)


_tune_malloc()


_SLOTS = 4
_CHUNK = 16 << 20  # 16 МБ

_ctx = pycuda.autoinit.context
_pin_in = [cuda.pagelocked_empty(_CHUNK // 4, np.float32) for _ in range(_SLOTS)]
_pin_out = [cuda.pagelocked_empty(_CHUNK // 4, np.float32) for _ in range(_SLOTS)]
_streams = [cuda.Stream() for _ in range(_SLOTS)]
_ev_done = [cuda.Event(cuda.event_flags.DISABLE_TIMING) for _ in range(_SLOTS)]

_state = {"bytes": 0, "d_in": None, "d_out": None, "cols": 0, "d_g": None, "d_b": None}


def _device_buffers(nbytes, cols):
    st = _state
    if nbytes > st["bytes"]:
        st["d_in"] = st["d_out"] = None  # освободить старые до выделения новых
        st["d_in"] = cuda.mem_alloc(nbytes)
        st["d_out"] = cuda.mem_alloc(nbytes)
        st["bytes"] = nbytes
    if cols > st["cols"]:
        st["d_g"] = cuda.mem_alloc(cols * 4)
        st["d_b"] = cuda.mem_alloc(cols * 4)
        st["cols"] = cols
    return st["d_in"], st["d_out"], st["d_g"], st["d_b"]


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32)
    b = np.ascontiguousarray(beta, dtype=np.float32)
    cols = int(row_size)
    rows = x.size // cols

    d_in, d_out, d_g, d_b = _device_buffers(x.nbytes, cols)
    cuda.memcpy_htod(d_g, g)
    cuda.memcpy_htod(d_b, b)
    kern, threads = _pick_kernel(cols)
    eps32, cols32 = np.float32(eps), np.int32(cols)
    out = np.empty_like(x)
    if cols * 4 > _CHUNK:
        cuda.memcpy_htod(d_in, x)
        kern.prepared_call((rows, 1), (threads, 1, 1), int(d_in), int(d_out),
                           d_g, d_b, cols32, eps32)
        cuda.memcpy_dtoh(out, d_out)
        return out

    r = min(rows, _CHUNK // (cols * 4))
    chunks = (rows + r - 1) // r
    issued = threading.Semaphore(0)      
    free = threading.Semaphore(_SLOTS)  

    def back(): 
        _ctx.push() 
        try:
            for i in range(chunks):
                issued.acquire()
                s, off = i % _SLOTS, i * r * cols
                sz = min(r, rows - i * r) * cols
                _ev_done[s].synchronize()
                np.copyto(out[off:off + sz], _pin_out[s][:sz])
                free.release()
        finally:
            _ctx.pop()

    t = threading.Thread(target=back)
    t.start()
    for i in range(chunks):
        s, off = i % _SLOTS, i * r * cols
        sz = min(r, rows - i * r) * cols

        free.acquire()
        np.copyto(_pin_in[s][:sz], x[off:off + sz])
        st = _streams[s]
        cuda.memcpy_htod_async(int(d_in) + off * 4, _pin_in[s][:sz], st)
        kern.prepared_async_call((sz // cols, 1), (threads, 1, 1), st,
                                 int(d_in) + off * 4, int(d_out) + off * 4,
                                 d_g, d_b, cols32, eps32)
        cuda.memcpy_dtoh_async(_pin_out[s][:sz], int(d_out) + off * 4, st)
        _ev_done[s].record(st)
        issued.release()
    t.join()
    return out
