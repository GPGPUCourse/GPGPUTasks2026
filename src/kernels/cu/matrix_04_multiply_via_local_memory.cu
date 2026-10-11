#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* __restrict__ a, // rows=h x cols=k
    const float* __restrict__ b, // rows=k x cols=w
    float* __restrict__ c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    constexpr unsigned int BM = 128;
    constexpr unsigned int BN = 64;
    constexpr unsigned int BK = 16;
    constexpr unsigned int THREADS = 16;
    __shared__ float a_tile[BM][BK];
    __shared__ float b_tile[BK][BN + 16];
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int row0 = blockIdx.y * BM + ty;
    const unsigned int col0 = blockIdx.x * BN + tx;

    float acc[8][4] = {};
    for (unsigned int k0 = 0; k0 < k; k0 += BK) {
#pragma unroll
        for (unsigned int r = 0; r < 8; ++r) {
            const unsigned int row = row0 + r * THREADS;
            a_tile[ty + r * THREADS][tx] = (row < h && k0 + tx < k) ? a[row * k + k0 + tx] : 0.0f;
        }
#pragma unroll
        for (unsigned int n = 0; n < 4; ++n) {
            const unsigned int col = col0 + n * THREADS;
            b_tile[ty][tx + n * THREADS] = (k0 + ty < k && col < w) ? b[(k0 + ty) * w + col] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (unsigned int i = 0; i < BK; ++i) {
            float av[8];
            float bv[4];
#pragma unroll
            for (unsigned int r = 0; r < 8; ++r)
                av[r] = a_tile[ty + r * THREADS][i];
#pragma unroll
            for (unsigned int n = 0; n < 4; ++n)
                bv[n] = b_tile[i][tx + n * THREADS];
#pragma unroll
            for (unsigned int r = 0; r < 8; ++r) {
#pragma unroll
                for (unsigned int n = 0; n < 4; ++n)
                    acc[r][n] = fmaf(av[r], bv[n], acc[r][n]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned int r = 0; r < 8; ++r) {
        const unsigned int row = row0 + r * THREADS;
        if (row < h) {
#pragma unroll
            for (unsigned int n = 0; n < 4; ++n) {
                const unsigned int col = col0 + n * THREADS;
                if (col < w)
                    c[row * w + col] = acc[r][n];
            }
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
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
