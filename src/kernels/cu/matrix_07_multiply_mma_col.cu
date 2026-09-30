#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cuda_fp16.h>

// 128x128 CTA, K-tile 32. A is column-major fp16, B is row-major fp16.
// A smem: ColumnMajorVoltaTensorOpMultiplicandCongruous, stride 128.
// B smem: RowMajorVoltaTensorOpMultiplicandBCongruous.

__host__ __device__ int offset_b_colmma(int row_k, int col_n)
{
    int vec_contiguous_idx = col_n >> 3;
    int tile_contiguous_residual = vec_contiguous_idx & 7;
    int tile_strided_residual = row_k & 3;
    int permuted_strided = tile_contiguous_residual & 3;
    int permuted_contiguous = (tile_strided_residual ^ permuted_strided) | (tile_contiguous_residual & 4);
    int element_contiguous = ((vec_contiguous_idx >> 3) * 8 + permuted_contiguous) * 8 + (col_n & 7);
    int element_strided = (row_k >> 2) * 4 + permuted_strided;
    return element_contiguous + element_strided * 128;
}

__host__ __device__ int offset_a_colmma(int row, int k)
{
    int vec_c = row >> 3;
    int res_c = vec_c & 7;
    int res_s = k & 3;
    int perm_s = res_c >> 1;
    int perm_c = (res_s ^ perm_s) | ((res_c & 1) << 2);
    int element_c = ((vec_c >> 3) * 8 + perm_c) * 8 + (row & 7);
    int element_s = (k >> 2) * 4 + perm_s;
    return element_c + element_s * 128;
}

__device__ __forceinline__ void mma_colrow(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tile_col(float acc[16][8], const unsigned a0[4], const unsigned a1[4],
                                             const unsigned b0[4], const unsigned b1[4])
{
#pragma unroll
    for (int outer_col = 0; outer_col < 2; ++outer_col)
#pragma unroll
        for (int inner_col = 0; inner_col < 2; ++inner_col)
#pragma unroll
            for (int outer_row = 0; outer_row < 2; ++outer_row)
#pragma unroll
                for (int inner_row = 0; inner_row < 2; ++inner_row) {
                    int op_col = inner_col + 2 * outer_col;
                    int inner_row_serp = inner_row;
                    int outer_row_serp = outer_row;
                    if (op_col & 1) {
                        inner_row_serp = 1 - inner_row;
                        outer_row_serp = 1 - outer_row;
                    }
                    int op_row = inner_row_serp + 2 * outer_row_serp;
                    int op_idx = inner_row_serp + 2 * (inner_col + 2 * (outer_row_serp + 2 * outer_col));
                    mma_colrow(acc[op_idx], a0[op_row], a1[op_row], b0[op_col], b1[op_col]);
                }
}

__device__ __forceinline__ void read_a_colmma(const uint4* ap0, const uint4* ap1, int a_byte, unsigned a0[4], unsigned a1[4])
{
    uint4 v0 = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(ap0) + a_byte);
    uint4 v1 = *reinterpret_cast<const uint4*>(reinterpret_cast<const char*>(ap1) + a_byte);
    a0[0] = v0.x;
    a1[0] = v0.y;
    a0[1] = v0.z;
    a1[1] = v0.w;
    a0[2] = v1.x;
    a1[2] = v1.y;
    a0[3] = v1.z;
    a1[3] = v1.w;
}

__device__ __forceinline__ void read_b_colmma(const uint4* bp, int b_byte, unsigned b0[4], unsigned b1[4])
{
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const char* p = reinterpret_cast<const char*>(bp + 4 * c) + b_byte;
        uint4 v = *reinterpret_cast<const uint4*>(p);
        b0[c * 2] = v.x;
        b1[c * 2] = v.y;
        b0[c * 2 + 1] = v.z;
        b1[c * 2 + 1] = v.w;
    }
}

__device__ __forceinline__ void load_tile_colmma(const half* A, const half* B, int block_m, int block_n, int k0, int w,
                                                 int h, int tid, uint4 preA[4], uint4 preB[4])
{
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        int vec = tid + rep * 128;
        int k_local = vec >> 4;
        int along = (vec & 15) << 3;
        preA[rep] = *reinterpret_cast<const uint4*>(A + ((size_t)(k0 + k_local) * h + (block_m + along)));
        preB[rep] = *reinterpret_cast<const uint4*>(B + ((size_t)(k0 + k_local) * w + (block_n + along)));
    }
}

__device__ __forceinline__ void store_tile_colmma(half (*As)[4096], half (*Bs)[4096], int stage, int tid,
                                                  const uint4 preA[4], const uint4 preB[4])
{
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        int vec = tid + rep * 128;
        int k_local = vec >> 4;
        int along = (vec & 15) << 3;
        *reinterpret_cast<uint4*>(&As[stage][offset_a_colmma(along, k_local)]) = preA[rep];
        *reinterpret_cast<uint4*>(&Bs[stage][offset_b_colmma(k_local, along)]) = preB[rep];
    }
}

__global__ void matrix_multiply_mma_col_tile(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
                                             int w, int h, int k)
{
    __shared__ __align__(16) half smem_h[4][4096];
    half (*As)[4096] = smem_h;
    half (*Bs)[4096] = smem_h + 2;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp & 1;
    const int warp_n = warp >> 1;
    const int block_m = blockIdx.y * 128;
    const int block_n = blockIdx.x * 128;

    float acc[16][8];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.f;

    uint4 preA[4];
    uint4 preB[4];

    int quad = lane >> 2;
    int lane_in_quad = lane & 3;
    int vec_row0 = lane >> 4;
    int vec_col = (lane & 4) >> 2;
    int vec_row1 = vec_row0 | 2;
    int a_c0 = (vec_col << 2) | ((lane & 3) ^ vec_row0);
    int a_s0 = vec_row0;
    int a_c1 = (vec_col << 2) | ((lane & 3) ^ vec_row1);
    int a_s1 = vec_row1;
    int b_access_strided = (lane >> 3) & 3;
    int b_access_contiguous = (lane ^ (lane >> 3)) & 3;
    int lane_m = (((quad & 4) >> 1) + (quad & 1)) * 8 + (lane_in_quad & 1);
    int lane_n = ((quad >> 1) & 1) * 8 + (lane_in_quad & 2);

    load_tile_colmma(A, B, block_m, block_n, 0, w, h, tid, preA, preB);
    store_tile_colmma(As, Bs, 0, tid, preA, preB);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < k; k0 += 32) {
        int next = k0 + 32;
        if (next < k)
            load_tile_colmma(A, B, block_m, block_n, next, w, h, tid, preA, preB);

        const uint4* base_a = reinterpret_cast<const uint4*>(As[stage]);
        const uint4* ap0 = base_a + a_c0 + a_s0 * 16;
        const uint4* ap1 = base_a + a_c1 + a_s1 * 16;
        int a_byte = warp_m * 128;
        const uint4* bp = reinterpret_cast<const uint4*>(Bs[stage]) + b_access_contiguous + b_access_strided * 16;
        int b_byte = warp_n * 64 * (int)sizeof(half);

        unsigned a0[2][4], a1[2][4], b0[2][4], b1[2][4];
        read_a_colmma(ap0, ap1, a_byte, a0[0], a1[0]);
        read_b_colmma(bp, b_byte, b0[0], b1[0]);
        a_byte += 1024;
        b_byte += 1024;
#pragma unroll
        for (int kk = 0; kk < 7; ++kk) {
            int nxt = (kk + 1) & 1;
            int cur = kk & 1;
            read_a_colmma(ap0, ap1, a_byte, a0[nxt], a1[nxt]);
            read_b_colmma(bp, b_byte, b0[nxt], b1[nxt]);
            a_byte += 1024;
            b_byte += 1024;
            mma_tile_col(acc, a0[cur], a1[cur], b0[cur], b1[cur]);
        }
        mma_tile_col(acc, a0[1], a1[1], b0[1], b1[1]);

        __syncthreads();
        if (next < k)
            store_tile_colmma(As, Bs, stage ^ 1, tid, preA, preB);
        __syncthreads();
        stage ^= 1;
    }

    float* smem = reinterpret_cast<float*>(smem_h);
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        int row_lo = pass * 64;
#pragma unroll
        for (int tile_n = 0; tile_n < 2; ++tile_n)
#pragma unroll
            for (int tile_m = 0; tile_m < 2; ++tile_m)
#pragma unroll
                for (int mma_n = 0; mma_n < 2; ++mma_n)
#pragma unroll
                    for (int mma_m = 0; mma_m < 2; ++mma_m) {
                        int group = ((tile_n * 2 + tile_m) * 2 + mma_n) * 2 + mma_m;
#pragma unroll
                        for (int p = 0; p < 2; ++p)
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                int accum_m = tile_m * 32 + mma_m * 4 + m * 2;
                                int accum_n = tile_n * 32 + mma_n * 4 + p * 16;
                                int idx = p * 4 + m * 2;
                                int local_row = warp_m * 64 + lane_m + accum_m;
                                int local_col = warp_n * 64 + lane_n + accum_n;
                                if (local_row >= row_lo && local_row < row_lo + 64) {
                                    int sr = local_row - row_lo;
                                    smem[sr * 128 + local_col] = acc[group][idx];
                                    smem[sr * 128 + local_col + 1] = acc[group][idx + 1];
                                }
                            }
                    }
        __syncthreads();
#pragma unroll
        for (int rep = 0; rep < 16; ++rep) {
            int elem = (tid + rep * 128) * 4;
            int row = elem >> 7;
            int col = elem & 127;
            float4 t = *reinterpret_cast<float4*>(smem + elem);
            *reinterpret_cast<float4*>(&C[((size_t)block_m + row_lo + row) * w + block_n + col]) = t;
        }
        __syncthreads();
    }
}

__global__ void f32_to_f16_rows_for_colmma(const float* __restrict__ in, half* __restrict__ out, size_t n)
{
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(in + i);
        reinterpret_cast<half2*>(out + i)[0] = __floats2half2_rn(v.x, v.y);
        reinterpret_cast<half2*>(out + i)[1] = __floats2half2_rn(v.z, v.w);
    }
}

__global__ void f32_rows_to_f16_cols(const float* __restrict__ in, half* __restrict__ out, int h, int k)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)h * k;
    if (i < n) {
        int row = (int)(i / (size_t)k);
        int col = (int)(i - (size_t)row * (size_t)k);
        out[(size_t)col * (size_t)h + row] = __float2half_rn(in[i]);
    }
}

namespace cuda {
void matrix_multiply_mma_col(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124314, context.type());
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
        if (cache_a.data)
            cudaFree(cache_a.data);
        cudaMalloc(&cache_a.data, na * sizeof(half));
        cache_a.cap = na;
        cache_a.src = nullptr;
    }
    if (!(cache_a.src == a.cuptr() && cache_a.n == na)) {
        int threads = 256;
        int blocks = (int)((na + threads - 1) / threads);
        f32_rows_to_f16_cols<<<blocks, threads, 0, stream>>>(a.cuptr(), cache_a.data, (int)h, (int)k);
        cache_a.src = a.cuptr();
        cache_a.n = na;
    }

    if (cache_b.cap < nb) {
        if (cache_b.data)
            cudaFree(cache_b.data);
        cudaMalloc(&cache_b.data, nb * sizeof(half));
        cache_b.cap = nb;
        cache_b.src = nullptr;
    }
    if (!(cache_b.src == b.cuptr() && cache_b.n == nb)) {
        int threads = 256;
        int blocks = (int)((nb / 4 + threads - 1) / threads);
        f32_to_f16_rows_for_colmma<<<blocks, threads, 0, stream>>>(b.cuptr(), cache_b.data, nb);
        cache_b.src = b.cuptr();
        cache_b.n = nb;
    }

    matrix_multiply_mma_col_tile<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        cache_a.data, cache_b.data, c.cuptr(), (int)w, (int)h, (int)k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
