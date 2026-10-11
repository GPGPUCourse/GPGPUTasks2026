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
    __shared__ half local_a[128][24];
    __shared__ half local_b[16][136];

    const int warp = threadIdx.x / 32;
    const int warpH = warp / 4;
    const int warpW = warp % 4;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(acc[i][j], 0);

    for (unsigned int i = 0; i < k; i += 16) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < 512; idx += 256) {
            const unsigned int y = idx / 4;
            const unsigned int x = (idx % 4) * 4;

            const float4 v = *reinterpret_cast<const float4*>(a + (blockIdx.y * 128 + y) * k + i + x);

            *reinterpret_cast<half2*>(&local_a[y][x]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<half2*>(&local_a[y][x + 2]) = __floats2half2_rn(v.z, v.w);
        }

        for (unsigned int idx = threadIdx.x; idx < 512; idx += 256) {
            const unsigned int y = idx / 32;
            const unsigned int x = (idx % 32) * 4;

            const float4 v = *reinterpret_cast<const float4*>(b + (i + y) * w + blockIdx.x * 128 + x);

            *reinterpret_cast<half2*>(&local_b[y][x]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<half2*>(&local_b[y][x + 2]) = __floats2half2_rn(v.z, v.w);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];

        #pragma unroll
        for (int j = 0; j < 4; ++j)
            wmma::load_matrix_sync(af[j], &local_a[(warpH * 4 + j) * 16][0], 24);

        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(bf[j], &local_b[0][(warpW * 2 + j) * 16], 136);

        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(c + (blockIdx.y * 128 + warpH * 64 + i * 16) * w + blockIdx.x * 128 + warpW * 32 + j * 16, acc[i][j], w, wmma::mem_row_major);
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
