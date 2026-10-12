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
    __shared__ float local_mem[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int i = blockIdx.x * TILE_SIZE + threadIdx.x;
    const unsigned int j = blockIdx.y * TILE_SIZE + threadIdx.y;

    if(i >= w || j >= h) {
        return;
    }

    local_mem[threadIdx.x][threadIdx.y] = matrix[j * w + i];
    
    __syncthreads();

    const unsigned int it = blockIdx.y * TILE_SIZE + threadIdx.x;
    const unsigned int jt = blockIdx.x * TILE_SIZE + threadIdx.y;
    transposed_matrix[jt * h + it] = local_mem[threadIdx.y][threadIdx.x];
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
