#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[16][16];
    const unsigned int tile_x = threadIdx.x;
    const unsigned int tile_y = threadIdx.y;
    if (x < w && y < h) {
        tile[tile_y][tile_x] = matrix[y * w + x];
    }

    __syncthreads();

    const unsigned int transposed_x = blockIdx.y * 16 + tile_x;
    const unsigned int transposed_y = blockIdx.x * 16 + tile_y;

    if (transposed_x >= h) {
        return;
    }
    if (transposed_y >= w) {
        return;
    }

    transposed_matrix[transposed_y * h + transposed_x] = tile[tile_x][tile_y];
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
