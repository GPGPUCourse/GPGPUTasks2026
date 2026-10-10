#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    __shared__ float local_a[TILE_SIZE][TILE_SIZE];
    __shared__ float local_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0;
    unsigned int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        unsigned int tiled_k = tile * TILE_SIZE + tx;
        unsigned int tiled_h = tile * TILE_SIZE + ty;

        // Загружаем блоки матрицы A в локальную память
        if (tiled_k < k && y < h) {
            local_a[ty][tx] = a[y * k + tiled_k];
        } else {
            local_a[ty][tx] = 0.0f;
        }

        // Загружаем блоки матрицы B в локальную память
        if (tiled_h < k && x < w) {
            local_b[ty][tx] = b[tiled_h * w + x];
        } else {
            local_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += local_a[ty][i] * local_b[i][tx];
        }

        __syncthreads();
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
