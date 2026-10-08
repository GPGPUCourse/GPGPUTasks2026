#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* __restrict__ a, // rows=h x cols=k
                       const float* __restrict__ b, // rows=k x cols=w
                             float* __restrict__ c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ half local_a[64][16];
    __shared__ half local_b[16][32];

    const int warp = threadIdx.x / 32;
    const int in_warp = threadIdx.x % 32;

    const int warp_row = warp / 2;
    const int warp_col = warp % 2;

    const unsigned int row = blockIdx.y * 64 + warp_row * 16;
    const unsigned int col = blockIdx.x * 32 + warp_col * 16;

    wmma::fragment<
        wmma::accumulator,
        16, 16, 16,
        float
    > acc;

    wmma::fill_fragment(acc, 0);

    for (unsigned int i = 0; i < k; i += 16)
    {
        {
            const int y = threadIdx.x / 4;
            const int x = (threadIdx.x % 4) * 4;
            
            const float4 v = *(float4*)(a + (blockIdx.y * 64 + y) * k + i + x);

            local_a[y][x] = __float2half(v.x);
            local_a[y][x + 1] = __float2half(v.y);
            local_a[y][x + 2] = __float2half(v.z);
            local_a[y][x + 3] = __float2half(v.w);
        }

        {
            const int y = threadIdx.x / 16;
            const int x = (threadIdx.x % 16) * 2;

            const float2 v = *(float2*)(b + (i + y) * w + blockIdx.x * 32 + x);

            local_b[y][x] = __float2half(v.x);
            local_b[y][x + 1] = __float2half(v.y);
        }

        __syncthreads();

        wmma::fragment<
            wmma::matrix_a,
            16, 16, 16,
            half,
            wmma::row_major
        > af;

        wmma::fragment<
            wmma::matrix_b,
            16, 16, 16,
            half,
            wmma::row_major
        > bf;

        wmma::load_matrix_sync(
            af,
            &local_a[warp_row * 16][0],
            16
        );

        wmma::load_matrix_sync(
            bf,
            &local_b[0][warp_col * 16],
            32
        );

        wmma::mma_sync(
            acc,
            af,
            bf,
            acc
        );

        __syncthreads();
    }

    wmma::store_matrix_sync(
        c + row * w + col,
        acc,
        w,
        wmma::mem_row_major
    );
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
