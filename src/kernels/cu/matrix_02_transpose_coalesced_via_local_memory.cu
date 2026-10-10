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

    const uint input_x = blockIdx.x * GROUP_SIZE_X_TRANSPOSE + threadIdx.x;
    const uint input_y = blockIdx.y * GROUP_SIZE_X_TRANSPOSE + threadIdx.y;

    __shared__ float local_data[GROUP_SIZE_X_TRANSPOSE][GROUP_SIZE_X_TRANSPOSE + 1];
    const uint local_x = threadIdx.x;
    const uint local_y = threadIdx.y;

    for (uint offset = 0; offset < GROUP_SIZE_X_TRANSPOSE; offset += GROUP_SIZE_Y_TRANSPOSE) {
        if (input_x < w && input_y + offset < h) {
            local_data[local_y + offset][local_x] = matrix[(input_y + offset) * w + input_x];
        }
    }
    __syncthreads();

    const uint output_x = blockIdx.y * GROUP_SIZE_X_TRANSPOSE + threadIdx.x;
    const uint output_y = blockIdx.x * GROUP_SIZE_X_TRANSPOSE + threadIdx.y;
    for (uint offset = 0; offset < GROUP_SIZE_X_TRANSPOSE; offset += GROUP_SIZE_Y_TRANSPOSE) {
        if (output_x < h && output_y + offset < w) {
            transposed_matrix[(output_y + offset) * h + output_x] = local_data[local_x][local_y + offset];
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
