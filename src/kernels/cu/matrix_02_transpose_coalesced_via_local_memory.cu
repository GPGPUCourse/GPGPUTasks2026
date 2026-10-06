#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

// works only for square work groups
__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* matrix, // w x h
    float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= h || column >= w) {
        return;
    }

    const unsigned int local_index = blockDim.x * threadIdx.y + (threadIdx.x + blockIdx.x) % blockDim.x;  // shift every row by blockIdx.x right (in cycle)
    const unsigned int src_index = row * w + column ;

    __shared__ float local_data[GROUP_SIZE];
    local_data[local_index] = matrix[src_index];
    __syncthreads();

    const unsigned int new_block_row = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int new_block_column = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int dst_index = new_block_row * h + new_block_column;
    const unsigned int local_index_of_data = blockDim.x * threadIdx.x + (threadIdx.y + blockIdx.x) % blockDim.x;

    transposed_matrix[dst_index] = local_data[local_index_of_data];
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
