#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    constexpr unsigned int TILE = 16;

    const unsigned int row0 = blockIdx.y * TILE;
    const unsigned int col0 = blockIdx.x * TILE;

    __shared__ half tile_a[TILE][TILE];
    __shared__ half tile_b[TILE][TILE];
    __shared__ float tile_c[TILE][TILE];

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> b_fragment;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_fragment;
    wmma::fill_fragment(c_fragment, 0.0f);

    const unsigned int lane = threadIdx.x;

    for (unsigned int k0 = 0; k0 < k; k0 += TILE) {
        for (unsigned int index = lane; index < TILE * TILE; index += 32) {
            const unsigned int local_row = index / TILE;
            const unsigned int local_col = index % TILE;

            const unsigned int a_row = row0 + local_row;
            const unsigned int a_col = k0 + local_col;
            const unsigned int b_row = k0 + local_row;
            const unsigned int b_col = col0 + local_col;

            tile_a[local_row][local_col] =
                (a_row < h && a_col < k) ? __float2half(a[a_row * k + a_col]) : __float2half(0.0f);
            tile_b[local_row][local_col] =
                (b_row < k && b_col < w) ? __float2half(b[b_row * w + b_col]) : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_fragment, &tile_a[0][0], TILE);
        wmma::load_matrix_sync(b_fragment, &tile_b[0][0], TILE);
        wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

        __syncthreads();
    }

    wmma::store_matrix_sync(&tile_c[0][0], c_fragment, TILE, wmma::mem_row_major);
    __syncthreads();

    for (unsigned int index = lane; index < TILE * TILE; index += 32) {
        const unsigned int local_row = index / TILE;
        const unsigned int local_col = index % TILE;
        const unsigned int row = row0 + local_row;
        const unsigned int col = col0 + local_col;
        if (row < h && col < w) {
            c[row * w + col] = tile_c[local_row][local_col];
        }
    }
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
