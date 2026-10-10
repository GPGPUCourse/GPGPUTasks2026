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
    // TODO 020 Это добровольное задание за супер-пупер-баллы престижа сверх нормы
    __shared__ half local_a[TILES_Y][TILE_SIZE][WGSIZE_X/TILE_SIZE][TILE_SIZE];
    __shared__ half local_b[WGSIZE_X/TILE_SIZE][TILES_X][TILE_SIZE][TILE_SIZE];

    uint32_t wi = threadIdx.y;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[TILES_X];
    for (int j = 0; j < TILES_X; ++j) {
      wmma::fill_fragment(c_frag[j], 0.0f);
    }

    for (int i = 0; i < k/(TILE_SIZE); i += 2) {
      for (int tile_y = 0; tile_y < TILES_Y; ++tile_y) {
        uint32_t offsetY1 = blockIdx.y * TILES_Y * TILE_SIZE + tile_y * TILE_SIZE + threadIdx.y;
        uint32_t offsetY2 = offsetY1 + WGSIZE_Y; // TILES_SIZE = 2 * WGSIZE_Y
        uint32_t offsetX = i * TILE_SIZE + threadIdx.x;
        half a1 = a[offsetY1 * k + offsetX];
        half a2 = a[offsetY2 * k + offsetX];
        local_a[tile_y][threadIdx.y][threadIdx.x/TILE_SIZE][threadIdx.x%TILE_SIZE] = a1; // load two tiles' rows at once
        local_a[tile_y][threadIdx.y + WGSIZE_Y][threadIdx.x/TILE_SIZE][threadIdx.x%TILE_SIZE] = a2; // load two tiles' rows at once
      }
      for (int tile_y = 0; tile_y < WGSIZE_X/TILE_SIZE; ++tile_y) {
        for (int tile_x = 0; tile_x < TILES_X; tile_x += 2) {
          uint32_t offsetX = blockIdx.x * TILES_X * TILE_SIZE + tile_x * TILE_SIZE + threadIdx.x;
          uint32_t offsetY1 = i * TILE_SIZE + tile_y * TILE_SIZE + threadIdx.y;
          uint32_t offsetY2 = offsetY1 + WGSIZE_Y; // TILES_SIZE = 2 * WGSIZE_Y
          half b1 = b[offsetY1 * w + offsetX];
          half b2 = b[offsetY2 * w + offsetX];
          local_b[tile_y][tile_x+threadIdx.x/TILE_SIZE][threadIdx.y][threadIdx.x%TILE_SIZE] = b1;
          local_b[tile_y][tile_x+threadIdx.x/TILE_SIZE][threadIdx.y+WGSIZE_Y][threadIdx.x%TILE_SIZE] = b2;
        }
      }
      __syncthreads();
      for (int tile_x = 0; tile_x < TILES_X; ++tile_x) {
        wmma::load_matrix_sync(a_frag, (half*)local_a[wi][0][0], WGSIZE_X);
        wmma::load_matrix_sync(b_frag, (half*)local_b[0][tile_x], 16);
        wmma::mma_sync(c_frag[tile_x], a_frag, b_frag, c_frag[tile_x]);
        wmma::load_matrix_sync(a_frag, (half*)local_a[wi][0][1], WGSIZE_X);
        wmma::load_matrix_sync(b_frag, (half*)local_b[1][tile_x], 16);
        wmma::mma_sync(c_frag[tile_x], a_frag, b_frag, c_frag[tile_x]);
      }
      __syncthreads();
    }

    uint32_t warpOffsetX = blockIdx.x * (TILE_SIZE * TILES_X);
    uint32_t warpOffsetY = blockIdx.y * (TILE_SIZE * TILES_Y) + wi * TILE_SIZE;
    for (int tile_x = 0; tile_x < TILES_X; ++tile_x) {
      wmma::store_matrix_sync(c + (warpOffsetY*w + warpOffsetX + tile_x * TILE_SIZE), c_frag[tile_x], w, wmma::mem_row_major);
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
