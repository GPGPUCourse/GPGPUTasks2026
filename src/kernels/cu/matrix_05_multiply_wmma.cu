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
    constexpr unsigned int threads = WARPS_H * WARPS_W * 32;
    constexpr unsigned int fragsH = BLOCK_H / (16 * WARPS_H);
    constexpr unsigned int fragsW = BLOCK_W / (16 * WARPS_W);

    __shared__ half local_a[BLOCK_H][BLOCK_K];
    __shared__ half local_b[BLOCK_K][BLOCK_W];

    const int warp = threadIdx.x / 32;

    const int warpH = warp / WARPS_W;
    const int warpW = warp % WARPS_W;

    const unsigned int H = blockIdx.y * BLOCK_H + warpH * 16 * fragsH;
    const unsigned int W = blockIdx.x * BLOCK_W + warpW * 16 * fragsW;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[fragsH][fragsW];

    #pragma unroll
    for (int i = 0; i < fragsH; ++i)
        #pragma unroll
        for (int j = 0; j < fragsW; ++j)
            wmma::fill_fragment(acc[i][j], 0);

    for (unsigned int i = 0; i < k; i += BLOCK_K) {
        for (unsigned int idx = threadIdx.x; idx < BLOCK_H * BLOCK_K; idx += threads) {
            const unsigned int y = idx / BLOCK_K;
            const unsigned int x = idx % BLOCK_K;

            local_a[y][x] = __float2half(a[(blockIdx.y * BLOCK_H + y) * k + i + x]);
        }

        for (unsigned int idx = threadIdx.x; idx < BLOCK_W * BLOCK_K; idx += threads) {
            const unsigned int y = idx / BLOCK_W;
            const unsigned int x = idx % BLOCK_W;

            local_b[y][x] = __float2half(b[(i + y) * w + blockIdx.x * BLOCK_W + x]);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[fragsH];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[fragsW];

        #pragma unroll
        for (int j = 0; j < fragsH; ++j)
            wmma::load_matrix_sync(af[j], &local_a[(warpH * fragsH + j) * 16][0], BLOCK_K);

        #pragma unroll
        for (int j = 0; j < fragsW; ++j)
            wmma::load_matrix_sync(bf[j], &local_b[0][(warpW * fragsW + j) * 16], BLOCK_W);

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
