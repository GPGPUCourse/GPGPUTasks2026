#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

template <int MODE>
__device__ __forceinline__ float4 ld_f4(const float* p)
{
    uint4 u;
    if constexpr (MODE == 1) {
        asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
                     : "l"(p));
    } else if constexpr (MODE == 2) {
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
                     : "l"(p));
    } else {
        u = *reinterpret_cast<const uint4*>(p);
    }
    float4 f;
    f.x = __uint_as_float(u.x);
    f.y = __uint_as_float(u.y);
    f.z = __uint_as_float(u.z);
    f.w = __uint_as_float(u.w);
    return f;
}

// BM x BN output tile, BK along K. Each thread owns TM x TN outputs.
// As is transposed so a whole warp broadcasts one A element.
template <int BM, int BN, int BK, int TM, int TN, bool STREAM>
__global__ void matrix_multiply_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int w,
    int h,
    int k)
{
    constexpr int THREADS = (BN / TN) * (BM / TM);
    __shared__ __align__(16) float As[BK][BM];
    __shared__ __align__(16) float Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int block_m = blockIdx.y * BM;
    const int block_n = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.f;

    for (int k0 = 0; k0 < k; k0 += BK) {
        constexpr int a_per = (BM * BK) / (4 * THREADS);
#pragma unroll
        for (int t = 0; t < a_per; ++t) {
            int elem = (tid + t * THREADS) * 4;
            int local_m = elem / BK;
            int local_k = elem - local_m * BK;
            const float* gp = A + (size_t)(block_m + local_m) * k + (k0 + local_k);
            float4 v;
            if constexpr (STREAM)
                v = ld_f4<1>(gp);
            else
                v = ld_f4<0>(gp);
            As[local_k + 0][local_m] = v.x;
            As[local_k + 1][local_m] = v.y;
            As[local_k + 2][local_m] = v.z;
            As[local_k + 3][local_m] = v.w;
        }

        constexpr int b_per = (BN * BK) / (4 * THREADS);
#pragma unroll
        for (int t = 0; t < b_per; ++t) {
            int elem = (tid + t * THREADS) * 4;
            int local_n = elem % BN;
            int local_k = elem / BN;
            const float* gp = B + (size_t)(k0 + local_k) * w + (block_n + local_n);
            float4 v;
            if constexpr (STREAM)
                v = ld_f4<2>(gp);
            else
                v = ld_f4<0>(gp);
            Bs[local_k][local_n + 0] = v.x;
            Bs[local_k][local_n + 1] = v.y;
            Bs[local_k][local_n + 2] = v.z;
            Bs[local_k][local_n + 3] = v.w;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM];
            float bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                av[i] = As[kk][ty + i * (BM / TM)];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                bv[j] = Bs[kk][tx + j * (BN / TN)];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_m + ty + i * (BM / TM);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = block_n + tx + j * (BN / TN);
            C[(size_t)row * w + col] = acc[i][j];
        }
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    matrix_multiply_tiled<128, 128, 16, 8, 8, true><<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        a.cuptr(), b.cuptr(), c.cuptr(), (int)w, (int)h, (int)k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
