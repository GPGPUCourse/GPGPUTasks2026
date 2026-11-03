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
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned tile_x = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    const unsigned tile_y = (blockIdx.y * blockDim.y + threadIdx.y) / 2;
    const unsigned wx = tile_x * WMMA_TILE_SIZE;
    const unsigned wy = tile_y * WMMA_TILE_SIZE;
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned warp = ty / 2;

    const unsigned count_tiles = GROUP_SIZE / 32;
    __shared__ half tile_a[count_tiles][WMMA_TILE_SIZE][WMMA_TILE_SIZE];
    __shared__ half tile_b[count_tiles][WMMA_TILE_SIZE][WMMA_TILE_SIZE];

    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_TILE_SIZE) {
        wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, half, wmma::row_major> b_frag;

        for (int y = 0; y < 8; ++y) {
            tile_a[warp][(ty % 2) * 8 + y][tx] = __float2half(a[(wy + (ty % 2) * 8 + y) * k + i + tx]);
            tile_b[warp][(ty % 2) * 8 + y][tx] = __float2half(b[(i + (ty % 2) * 8 + y) * w + wx + tx]);
        }

        __syncwarp();

        wmma::load_matrix_sync(a_frag, &tile_a[warp][0][0], WMMA_TILE_SIZE);
        wmma::load_matrix_sync(b_frag, &tile_b[warp][0][0], WMMA_TILE_SIZE);

        __syncwarp();

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(c + wy * w + wx, c_frag, w, wmma::mem_row_major);
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
