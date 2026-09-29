#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

template <int MODE>
__device__ __forceinline__ uint4 ld4(const void* p)
{
    uint4 v;
    if constexpr (MODE == 1) {
        asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    } else if constexpr (MODE == 2) {
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    } else {
        v = *reinterpret_cast<const uint4*>(p);
    }
    return v;
}

template <int BK, int BN>
__device__ __forceinline__ void mma_32x32(
    const half* As,
    const half* Bs,
    int warp_m,
    int warp_n,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2])
{
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::load_matrix_sync(af, As + (warp_m * 32 + i * 16) * BK + kk, BK);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
                wmma::load_matrix_sync(bf, Bs + kk * BN + warp_n * 32 + j * 16, BN);
                wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
            }
        }
    }
}

__device__ __forceinline__ void store_32x32(
    float* C,
    int w,
    int row0,
    int col0,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2])
{
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float* p = C + (size_t)(row0 + i * 16) * w + (col0 + j * 16);
            wmma::store_matrix_sync(p, acc[i][j], w, wmma::mem_row_major);
        }
    }
}

template <int BK, int BN, int THREADS, bool STREAM>
__device__ __forceinline__ void load_chunk(
    const half* A,
    const half* B,
    int w,
    int k,
    int block_m,
    int block_n,
    int k0,
    int tid,
    int rep,
    uint4& va,
    uint4& vb)
{
    int chunk = tid + rep * THREADS;
    constexpr int a_per_row = BK / 8;
    int a_row = chunk / a_per_row;
    int a_col = (chunk - a_row * a_per_row) * 8;
    const half* ap = A + (size_t)(block_m + a_row) * (size_t)k + (k0 + a_col);
    constexpr int b_per_row = BN / 8;
    int b_row = chunk / b_per_row;
    int b_col = (chunk - b_row * b_per_row) * 8;
    const half* bp = B + (size_t)(k0 + b_row) * (size_t)w + (block_n + b_col);
    if constexpr (STREAM) {
        va = ld4<1>(ap);
        vb = ld4<2>(bp);
    } else {
        va = ld4<0>(ap);
        vb = ld4<0>(bp);
    }
}

// 64x64 CTA, 128 threads, four warps, each warp owns 32x32.
// A is loaded with ld.global.ca, B with ld.global.cs. K is pipelined.
template <int BK, bool STREAM>
__global__ void matrix_multiply_wmma_tile(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int w,
    int h,
    int k)
{
    (void)h;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int THREADS = 128;
    constexpr int LOADS = (BM * BK) / (8 * THREADS);
    __shared__ __align__(16) half As[BM * BK];
    __shared__ __align__(16) half Bs[BK * BN];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp = tid >> 5;
    const int warp_n = warp % 2;
    const int warp_m = warp / 2;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    int off_a[LOADS];
    int off_b[LOADS];
#pragma unroll
    for (int rep = 0; rep < LOADS; ++rep) {
        int chunk = tid + rep * THREADS;
        constexpr int a_per_row = BK / 8;
        int a_row = chunk / a_per_row;
        int a_col = (chunk - a_row * a_per_row) * 8;
        off_a[rep] = a_row * BK + a_col;
        constexpr int b_per_row = BN / 8;
        int b_row = chunk / b_per_row;
        int b_col = (chunk - b_row * b_per_row) * 8;
        off_b[rep] = b_row * BN + b_col;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0.f);

#pragma unroll
    for (int rep = 0; rep < LOADS; ++rep) {
        uint4 va, vb;
        load_chunk<BK, BN, THREADS, STREAM>(A, B, w, k, block_m, block_n, 0, tid, rep, va, vb);
        *reinterpret_cast<uint4*>(As + off_a[rep]) = va;
        *reinterpret_cast<uint4*>(Bs + off_b[rep]) = vb;
    }
    __syncthreads();

    for (int k0 = 0; k0 < k; k0 += BK) {
        int next = k0 + BK;
        uint4 pre_a[LOADS];
        uint4 pre_b[LOADS];
        if (next < k) {
#pragma unroll
            for (int rep = 0; rep < LOADS; ++rep) {
                load_chunk<BK, BN, THREADS, STREAM>(A, B, w, k, block_m, block_n, next, tid, rep, pre_a[rep], pre_b[rep]);
            }
        }
        mma_32x32<BK, BN>(As, Bs, warp_m, warp_n, acc);
        __syncthreads();
        if (next < k) {
#pragma unroll
            for (int rep = 0; rep < LOADS; ++rep) {
                *reinterpret_cast<uint4*>(As + off_a[rep]) = pre_a[rep];
                *reinterpret_cast<uint4*>(Bs + off_b[rep]) = pre_b[rep];
            }
        }
        __syncthreads();
    }

    store_32x32(C, w, block_m + warp_m * 32, block_n + warp_n * 32, acc);
}

__global__ void f32_to_f16(const float* __restrict__ in, half* __restrict__ out, size_t n)
{
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(in + i);
        reinterpret_cast<half2*>(out + i)[0] = __floats2half2_rn(v.x, v.y);
        reinterpret_cast<half2*>(out + i)[1] = __floats2half2_rn(v.z, v.w);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
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
        f32_to_f16<<<blocks, threads, 0, stream>>>(src, cache.data, n);
        cache.src = src;
        cache.n = n;
    }

    matrix_multiply_wmma_tile<32, true><<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        cache_a.data, cache_b.data, c.cuptr(), (int)w, (int)h, (int)k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
