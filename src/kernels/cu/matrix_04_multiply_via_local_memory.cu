#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=H x cols=K
                       const float* b, // rows=K x cols=W
                             float* c, // rows=H x cols=W
                       unsigned int W,
                       unsigned int H,
                       unsigned int K)
{
    float acc = 0;

    const unsigned tile_sz = 32;
    __shared__ float local_a[tile_sz][tile_sz];
    __shared__ float local_b[tile_sz][tile_sz];

    // Число тайлов считаем с округлением вверх, чтобы покрыть всю K
    unsigned num_tiles = (K + tile_sz - 1) / tile_sz;

    const unsigned glob_a_i = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned glob_b_j = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned t = 0; t < num_tiles; t++) {
        // Загрузка тайла в локальную память
        const unsigned glob_a_j = threadIdx.x + t * tile_sz;
        const unsigned glob_b_i = threadIdx.y + t * tile_sz;

        if (glob_a_j < K && glob_a_i < H)
            local_a[threadIdx.y][threadIdx.x] = a[glob_a_j + K * glob_a_i];
        else
            local_a[threadIdx.y][threadIdx.x] = 0;

        if (glob_b_j < W && glob_b_i < K)
            local_b[threadIdx.y][threadIdx.x] = b[glob_b_j + W * glob_b_i];
        else
            local_b[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Вычисления
        const unsigned local_j = threadIdx.x;
        const unsigned local_i = threadIdx.y;

        for (unsigned local_k = 0; local_k < tile_sz; local_k++)
            acc += local_a[local_i][local_k] * local_b[local_k][local_j];

        __syncthreads();
    }
    // Запись результата
    const unsigned glob_с_j = glob_b_j;
    const unsigned glob_с_i = glob_a_i;

    if (glob_с_j < W && glob_с_i < H)
        c[glob_с_j + glob_с_i * W] = acc;
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
