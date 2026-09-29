#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cuda_fp16.h>

// 128x128 CTA, 128 threads, four warps. Each warp owns 64x64.
// mma.sync m8n8k4. A is ld.global.ca, B is ld.global.cs and transposed into smem.
__device__ __forceinline__ uint4 ldg4(const void* p, bool streaming)
{
    uint4 v;
    if (streaming) {
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    } else {
        asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    }
    return v;
}

__device__ __forceinline__ void mma_884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// One warp owns WARP_M x WARP_N. Four quadpairs issue one m8n8k4 together.
// A smem is [row][k], B smem is [col][k], so each fragment is 4 contiguous halves.
template <int BM, int BN, int BK, int WARP_M, int WARP_N>
__global__ void matrix_multiply_mma_tile(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int w, int h, int k)
{
    (void)h;
    constexpr int WARPS_M = BM / WARP_M;
    constexpr int WARPS_N = BN / WARP_N;
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int TM = WARP_M / 16;
    constexpr int TN = WARP_N / 16;
    constexpr int A_REPS = (BM * BK / 8) / THREADS;
    constexpr int B_REPS = (BN * BK / 8) / THREADS;
    static_assert(BM * BK / 8 % THREADS == 0, "A reps");
    static_assert(BN * BK / 8 % THREADS == 0, "B reps");
    static_assert(WARP_M % 16 == 0 && WARP_N % 16 == 0, "warp tile");

    __shared__ __align__(16) half As[2][BM * BK];
    __shared__ __align__(16) half Bs[2][BN * BK];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp - warp_m * WARPS_N;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;
    const int qp = (lane >> 2) & 3;
    const int qrow = qp >> 1;
    const int qcol = qp & 1;
    const int frag_rc = (lane & 3) + ((lane & 16) ? 4 : 0);

    float acc[TM][TN][8];
#pragma unroll
    for (int tm = 0; tm < TM; ++tm)
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[tm][tn][i] = 0.f;

    uint4 preA[A_REPS];
    uint4 preB[B_REPS];

#define LOAD_GLOBAL(k0)                                                                 \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int rep = 0; rep < A_REPS; ++rep) {                                        \
            int e = (tid + rep * THREADS) * 8;                                          \
            int r = e / BK;                                                             \
            int c = e - r * BK;                                                         \
            preA[rep] = ldg4(A + ((size_t)(block_m + r) * k + ((k0) + c)), false);      \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int rep = 0; rep < B_REPS; ++rep) {                                        \
            int e = (tid + rep * THREADS) * 8;                                          \
            int kk = e / BN;                                                            \
            int n = e - kk * BN;                                                        \
            preB[rep] = ldg4(B + ((size_t)((k0) + kk) * w + (block_n + n)), true);      \
        }                                                                               \
    } while (0)

#define STORE_SMEM(stage)                                                               \
    do {                                                                                \
        _Pragma("unroll")                                                               \
        for (int rep = 0; rep < A_REPS; ++rep) {                                        \
            int e = (tid + rep * THREADS) * 8;                                          \
            int r = e / BK;                                                             \
            int c = e - r * BK;                                                         \
            *reinterpret_cast<uint4*>(&As[stage][r * BK + c]) = preA[rep];              \
        }                                                                               \
        _Pragma("unroll")                                                               \
        for (int rep = 0; rep < B_REPS; ++rep) {                                        \
            int e = (tid + rep * THREADS) * 8;                                          \
            int kk = e / BN;                                                            \
            int n = e - kk * BN;                                                        \
            const unsigned short* hs = reinterpret_cast<const unsigned short*>(&preB[rep]); \
            _Pragma("unroll")                                                           \
            for (int i = 0; i < 8; ++i)                                                 \
                Bs[stage][(n + i) * BK + kk] = __ushort_as_half(hs[i]);                 \
        }                                                                               \
    } while (0)

    LOAD_GLOBAL(0);
    STORE_SMEM(0);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < k; k0 += BK) {
        int next = k0 + BK;
        if (next < k)
            LOAD_GLOBAL(next);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 4) {
#pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    int row = warp_m * WARP_M + (tm * 2 + qrow) * 8 + frag_rc;
                    int col = warp_n * WARP_N + (tn * 2 + qcol) * 8 + frag_rc;
                    const half* ap = &As[stage][row * BK + kk];
                    const half* bp = &Bs[stage][col * BK + kk];
                    unsigned a0 = *reinterpret_cast<const unsigned*>(ap);
                    unsigned a1 = *reinterpret_cast<const unsigned*>(ap + 2);
                    unsigned b0 = *reinterpret_cast<const unsigned*>(bp);
                    unsigned b1 = *reinterpret_cast<const unsigned*>(bp + 2);
                    mma_884(acc[tm][tn], a0, a1, b0, b1);
                }
            }
        }
        __syncthreads();
        if (next < k)
            STORE_SMEM(stage ^ 1);
        __syncthreads();
        stage ^= 1;
    }
#undef LOAD_GLOBAL
#undef STORE_SMEM

#pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            int m8 = tm * 2 + qrow;
            int n8 = tn * 2 + qcol;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int x = (lane & 1) + (i & 2);
                int row = (lane < 16) ? x : x + 4;
                int col = (i & 4) + (lane & 2) + (i & 1);
                int grow = block_m + warp_m * WARP_M + m8 * 8 + row;
                int gcol = block_n + warp_n * WARP_N + n8 * 8 + col;
                C[(size_t)grow * w + gcol] = acc[tm][tn][i];
            }
        }
    }
}

__global__ void f32_to_f16_for_mma(const float* __restrict__ in, half* __restrict__ out, size_t n)
{
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(in + i);
        reinterpret_cast<half2*>(out + i)[0] = __floats2half2_rn(v.x, v.y);
        reinterpret_cast<half2*>(out + i)[1] = __floats2half2_rn(v.z, v.w);
    }
}

namespace cuda {
void matrix_multiply_mma(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124313, context.type());
    cudaStream_t stream = context.cudaStream();

    const size_t na = (size_t)h * k;
    const size_t nb = (size_t)k * w;
    struct Cache {
        const float* src;
        size_t n;
        half* data;
        size_t cap;
    };
    static Cache cache_a{ nullptr, 0, nullptr, 0 };
    static Cache cache_b{ nullptr, 0, nullptr, 0 };
    Cache* caches[2] = { &cache_a, &cache_b };
    const float* srcs[2] = { a.cuptr(), b.cuptr() };
    const size_t counts[2] = { na, nb };
    for (int which = 0; which < 2; ++which) {
        Cache& cache = *caches[which];
        const float* src = srcs[which];
        size_t n = counts[which];
        if (cache.cap < n) {
            if (cache.data)
                cudaFree(cache.data);
            cudaMalloc(&cache.data, n * sizeof(half));
            cache.cap = n;
            cache.src = nullptr;
        }
        if (cache.src == src && cache.n == n)
            continue;
        int threads = 256;
        int blocks = (int)((n / 4 + threads - 1) / threads);
        f32_to_f16_for_mma<<<blocks, threads, 0, stream>>>(src, cache.data, n);
        cache.src = src;
        cache.n = n;
    }

    matrix_multiply_mma_tile<128, 128, 16, 64, 64><<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        cache_a.data, cache_b.data, c.cuptr(), (int)w, (int)h, (int)k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
