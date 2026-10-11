#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

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
    __shared__ half local_b[16][64];

    const int warp = threadIdx.x / 32;

    const int warp_row = warp / 2;
    const int warp_col = warp % 2;

    const unsigned int row = blockIdx.y * 64 + warp_row * 16;
    const unsigned int col = blockIdx.x * 64 + warp_col * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];

    wmma::fill_fragment(acc[0], 0);
    wmma::fill_fragment(acc[1], 0);

    for (unsigned int i = 0; i < k; i += 16) {
        {
            const int y = threadIdx.x / 4;
            const int x = (threadIdx.x % 4) * 4;

            const float4 v = *reinterpret_cast<const float4*>(a + (blockIdx.y * 64 + y) * k + i + x);

            *reinterpret_cast<half2*>(&local_a[y][x]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<half2*>(&local_a[y][x + 2]) = __floats2half2_rn(v.z, v.w);
        }

        {
            const int y = threadIdx.x / 16;
            const int x = (threadIdx.x % 16) * 4;

            const float4 v = *reinterpret_cast<const float4*>(b + (i + y) * w + blockIdx.x * 64 + x);

            *reinterpret_cast<half2*>(&local_b[y][x]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<half2*>(&local_b[y][x + 2]) = __floats2half2_rn(v.z, v.w);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];

        wmma::load_matrix_sync(af, &local_a[warp_row * 16][0], 16);
        wmma::load_matrix_sync(bf[0], &local_b[0][warp_col * 32], 64);
        wmma::load_matrix_sync(bf[1], &local_b[0][warp_col * 32 + 16], 64);

        wmma::mma_sync(acc[0], af, bf[0], acc[0]);
        wmma::mma_sync(acc[1], af, bf[1], acc[1]);

        __syncthreads();
    }

    wmma::store_matrix_sync(c + row * w + col, acc[0], w, wmma::mem_row_major);
    wmma::store_matrix_sync(c + row * w + col + 16, acc[1], w, wmma::mem_row_major);
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
