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
    unsigned int i = blockIdx.x * 32 + threadIdx.x;
    unsigned int j = blockIdx.y * 32 + threadIdx.y;
    const unsigned int local_i = threadIdx.x;
    const unsigned int local_j = threadIdx.y;

    __shared__ float temp_matrix[32][33];

    for (unsigned int k = 0; k < 32; k += 8) {
        if (i < w && j + k < h) {
            temp_matrix[local_j + k][local_i] = matrix[(j + k) * w + i];
        }
    }
    __syncthreads();

    i = blockIdx.y * 32 + threadIdx.x;
    j = blockIdx.x * 32 + threadIdx.y;

    for (unsigned int k = 0; k < 32; k += 8) {
        if (i < h && j + k < w) {
            transposed_matrix[(j + k) * h + i] = temp_matrix[local_i][local_j + k];
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
