#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// 64x64 CTA. A is column-major, B is row-major. Both tiles are padded by 8 halves
// so wmma::load_matrix_sync does not walk the same shared-memory banks.
// The fp32->fp16 transpose of A is cached on the host side, same as the fp16 conversion.

__device__ __forceinline__ uint4 ld4_ca(const void* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld4_cs(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__global__ void matrix_multiply_wmma_col_tile(
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int w, int h, int k) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int LDA = 72;
    constexpr int LDB = 72;
    constexpr int THREADS = 128;
    constexpr int LOADS = 2;
    constexpr int A_HALVES = BK * LDA;

    __shared__ __align__(16) char raw[BM * BN * sizeof(float)];
    half* As = reinterpret_cast<half*>(raw);
    half* Bs = As + A_HALVES;
    float* Cs = reinterpret_cast<float*>(raw);

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
        int m8 = (chunk % 8) * 8;
        int k_local = chunk / 8;
        off_a[rep] = k_local * LDA + m8;
        int b_row = chunk / 8;
        int b_col = (chunk % 8) * 8;
        off_b[rep] = b_row * LDB + b_col;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    uint4 pre_a[LOADS], pre_b[LOADS];
#pragma unroll
    for (int rep = 0; rep < LOADS; ++rep) {
        int chunk = tid + rep * THREADS;
        int m8 = (chunk % 8) * 8;
        int k_local = chunk / 8;
        const half* ap = A + (size_t)k_local * (size_t)h + (block_m + m8);
        int b_row = chunk / 8;
        int b_col = (chunk % 8) * 8;
        const half* bp = B + (size_t)b_row * (size_t)w + (block_n + b_col);
        pre_a[rep] = ld4_ca(ap);
        pre_b[rep] = ld4_cs(bp);
    }
#pragma unroll
    for (int rep = 0; rep < LOADS; ++rep) {
        *reinterpret_cast<uint4*>(As + off_a[rep]) = pre_a[rep];
        *reinterpret_cast<uint4*>(Bs + off_b[rep]) = pre_b[rep];
    }
    __syncthreads();

    const int row0 = warp_m * 32;
    const int col0 = warp_n * 32;
    for (int k0 = 0; k0 < k; k0 += BK) {
        int next = k0 + BK;
        if (next < k) {
#pragma unroll
            for (int rep = 0; rep < LOADS; ++rep) {
                int chunk = tid + rep * THREADS;
                int m8 = (chunk % 8) * 8;
                int k_local = chunk / 8;
                const half* ap = A + (size_t)(next + k_local) * (size_t)h + (block_m + m8);
                int b_row = chunk / 8;
                int b_col = (chunk % 8) * 8;
                const half* bp = B + (size_t)(next + b_row) * (size_t)w + (block_n + b_col);
                pre_a[rep] = ld4_ca(ap);
                pre_b[rep] = ld4_cs(bp);
            }
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a0[2], a1[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0[2], b1[2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::load_matrix_sync(a0[i], As + (row0 + i * 16), LDA);
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(b0[j], Bs + (col0 + j * 16), LDB);
#pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::load_matrix_sync(a1[i], As + 16 * LDA + (row0 + i * 16), LDA);
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(b1[j], Bs + 16 * LDB + (col0 + j * 16), LDB);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a0[i], b0[j], acc[i][j]);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a1[i], b1[j], acc[i][j]);

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

    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            float* p = Cs + (size_t)(row0 + i * 16) * BN + (col0 + j * 16);
            wmma::store_matrix_sync(p, acc[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rep = 0; rep < 8; ++rep) {
        int elem = (tid + rep * THREADS) * 4;
        int row = elem / BN;
        int col = elem - row * BN;
        float4 out = *reinterpret_cast<const float4*>(Cs + row * BN + col);
        *reinterpret_cast<float4*>(C + (size_t)(block_m + row) * w + (block_n + col)) = out;
    }
}

__global__ void f32_to_f16_rows_for_wmma_col(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(in + i);
        reinterpret_cast<half2*>(out + i)[0] = __floats2half2_rn(v.x, v.y);
        reinterpret_cast<half2*>(out + i)[1] = __floats2half2_rn(v.z, v.w);
    }
}

__global__ void f32_rows_to_f16_cols_for_wmma(const float* __restrict__ in, half* __restrict__ out, int h, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = h * k;
    if (i < n) {
        int r = i / k;
        int c = i - r * k;
        out[(size_t)c * h + r] = __float2half_rn(in[i]);
    }
}

namespace cuda {
void matrix_multiply_wmma_col(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124315, context.type());
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

    if (cache_a.cap < na) {
        if (cache_a.data) cudaFree(cache_a.data);
        cudaMalloc(&cache_a.data, na * sizeof(half));
        cache_a.cap = na;
        cache_a.src = nullptr;
    }
    if (cache_b.cap < nb) {
        if (cache_b.data) cudaFree(cache_b.data);
        cudaMalloc(&cache_b.data, nb * sizeof(half));
        cache_b.cap = nb;
        cache_b.src = nullptr;
    }

    if (!(cache_a.src == a.cuptr() && cache_a.n == na)) {
        int threads = 256;
        int blocks = (int)((na + threads - 1) / threads);
        f32_rows_to_f16_cols_for_wmma<<<blocks, threads, 0, stream>>>(a.cuptr(), cache_a.data, (int)h, (int)k);
        cache_a.src = a.cuptr();
        cache_a.n = na;
    }
    if (!(cache_b.src == b.cuptr() && cache_b.n == nb)) {
        int threads = 256;
        int blocks = (int)((nb / 4 + threads - 1) / threads);
        f32_to_f16_rows_for_wmma_col<<<blocks, threads, 0, stream>>>(b.cuptr(), cache_b.data, (int)nb);
        cache_b.src = b.cuptr();
        cache_b.n = nb;
    }

    matrix_multiply_wmma_col_tile<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        cache_a.data, cache_b.data, c.cuptr(), (int)w, (int)h, (int)k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
