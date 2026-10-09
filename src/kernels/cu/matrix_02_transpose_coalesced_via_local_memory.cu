#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

#include <iostream>

__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* matrix, // w x h
    float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    const unsigned int bank_bit_count = 5;

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int local_index = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float local_data[GROUP_SIZE + (GROUP_SIZE >> bank_bit_count)];

    if (x < w && y < h)
        local_data[local_index + (local_index >> bank_bit_count)] = matrix[x + y * w];
    __syncthreads();

    const unsigned int new_x = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int new_y = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int new_local_index = threadIdx.x * blockDim.x + threadIdx.y;

    if (x < w && y < h)
        transposed_matrix[new_x + new_y * h] = local_data[new_local_index + (new_local_index >> bank_bit_count)];
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
