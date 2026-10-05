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

    __shared__ float local_data[GROUP_SIZE_S][GROUP_SIZE_S + 1];

    const unsigned int src_y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int src_x = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int dst_y = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned int dst_x = blockIdx.y * blockDim.y + threadIdx.x;

    local_data[threadIdx.y][threadIdx.x] = matrix[src_y * w + src_x];

    __syncthreads();

    transposed_matrix[dst_y * h + dst_x] = local_data[threadIdx.x][threadIdx.y];
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
