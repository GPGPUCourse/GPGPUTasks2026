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
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int col_t = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int row_t = blockIdx.x * blockDim.x + threadIdx.y;

    const unsigned int local_index = threadIdx.y * (GROUP_SIZE_X + 1) + threadIdx.x;
    const unsigned int transposed_local_index = threadIdx.x * (GROUP_SIZE_X + 1) + threadIdx.y;

    __shared__ float local_data[GROUP_SIZE_Y * (GROUP_SIZE_X + 1)];

    if (col < w && row < h) {
        local_data[local_index] = matrix[row * w + col];
    }
    __syncthreads();
    if (col_t < h && row_t < w) {
        transposed_matrix[row_t * h + col_t] = local_data[transposed_local_index];
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
