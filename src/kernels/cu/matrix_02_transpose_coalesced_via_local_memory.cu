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
    const int TILE_SIZE = 32;
    unsigned int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    unsigned int local_x = threadIdx.x;
    unsigned int local_y = threadIdx.y;


    __shared__ float local_data[TILE_SIZE][TILE_SIZE + 1];

    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        if (y + i < h and x < w) {
            local_data[(local_y + i)][local_x] = matrix[(y + i) * w + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int i = 0; i < TILE_SIZE; i += blockDim.y) {
        if (y + i < w and x < h) {

            transposed_matrix[(y + i) * h + x] = local_data[local_x][local_y + i];
        }
    }

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
