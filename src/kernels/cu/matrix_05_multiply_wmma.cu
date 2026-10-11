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
    constexpr unsigned int blockH = 128;
    constexpr unsigned int blockK = 16;
    constexpr unsigned int blockW = 128;
    constexpr unsigned int warpsH = 2;
    constexpr unsigned int warpsW = 4;

    constexpr unsigned int threads = warpsH * warpsW * 32;
    constexpr unsigned int fragsH = blockH / (16 * warpsH);
    constexpr unsigned int fragsW = blockW / (16 * warpsW);

    __shared__ half local_a[blockH][blockK + 8];
    __shared__ half local_b[blockK][blockW + 8];

    const int warp = threadIdx.x / 32;

    const int warpH = warp / warpsW;
    const int warpW = warp % warpsW;

    const unsigned int H = blockIdx.y * blockH + warpH * 16 * fragsH;
    const unsigned int W = blockIdx.x * blockW + warpW * 16 * fragsW;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[fragsH][fragsW];

    #pragma unroll
    for (int i = 0; i < fragsH; ++i)
        #pragma unroll
        for (int j = 0; j < fragsW; ++j)
            wmma::fill_fragment(acc[i][j], 0);

    for (unsigned int i = 0; i < k; i += blockK) {
        for (unsigned int idx = threadIdx.x; idx < blockH * (blockK / 4); idx += threads) {
            const unsigned int y = idx / (blockK / 4);
            const unsigned int x = (idx % (blockK / 4)) * 4;

            const float4 v = *reinterpret_cast<const float4*>(a + (blockIdx.y * blockH + y) * k + i + x);

            *reinterpret_cast<half2*>(&local_a[y][x]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<half2*>(&local_a[y][x + 2]) = __floats2half2_rn(v.z, v.w);
        }

        for (unsigned int idx = threadIdx.x; idx < blockK * (blockW / 4); idx += threads) {
            const unsigned int y = idx / (blockW / 4);
            const unsigned int x = (idx % (blockW / 4)) * 4;

            const float4 v = *reinterpret_cast<const float4*>(b + (i + y) * w + blockIdx.x * blockW + x);

            *reinterpret_cast<half2*>(&local_b[y][x]) = __floats2half2_rn(v.x, v.y);
            *reinterpret_cast<half2*>(&local_b[y][x + 2]) = __floats2half2_rn(v.z, v.w);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[fragsH];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[fragsW];

        #pragma unroll
        for (int j = 0; j < fragsH; ++j)
            wmma::load_matrix_sync(af[j], &local_a[(warpH * fragsH + j) * 16][0], blockK + 8);

        #pragma unroll
        for (int j = 0; j < fragsW; ++j)
            wmma::load_matrix_sync(bf[j], &local_b[0][(warpW * fragsW + j) * 16], blockW + 8);

        #pragma unroll
        for (int i = 0; i < fragsH; ++i)
            #pragma unroll
            for (int j = 0; j < fragsW; ++j)
                wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < fragsH; ++i)
        #pragma unroll
        for (int j = 0; j < fragsW; ++j)
            wmma::store_matrix_sync(c + (H + i * 16) * w + W + j * 16, acc[i][j], w, wmma::mem_row_major);
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
