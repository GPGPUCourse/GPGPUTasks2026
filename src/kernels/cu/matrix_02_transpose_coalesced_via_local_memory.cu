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
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float temp[GROUP_SIZE_Y * (GROUP_SIZE_X + 1)];
    if (x < w && y < h) {
        temp[threadIdx.y * (GROUP_SIZE_X + 1) + threadIdx.x] = matrix[y * w + x];
    }
    __syncthreads();
    unsigned int new_x = blockIdx.y * blockDim.y + threadIdx.x;
    unsigned int new_y = blockIdx.x * blockDim.x + threadIdx.y;
    if (new_x < h && new_y < w) {
        transposed_matrix[new_y * h + new_x] = temp[threadIdx.x * (GROUP_SIZE_X + 1) + threadIdx.y];
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
