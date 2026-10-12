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
    int idx = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    int idy = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;

    __shared__ float shared_matrix[GROUP_SIZE_Y][GROUP_SIZE_X + 1]; //padded matrix

    for (int i = 0; i < GROUP_SIZE_Y; i += blockDim.y) {
        if (idx < w && (idy + i) < h) shared_matrix[threadIdx.y + i][threadIdx.x] = matrix[(idy + i) * w + idx];
    }
    __syncthreads();

    idx = blockIdx.y * GROUP_SIZE_Y + threadIdx.x;
    idy = blockIdx.x * GROUP_SIZE_X + threadIdx.y;

    for (int i = 0; i < GROUP_SIZE_Y; i += blockDim.y) {
         if (idx < h && (idy + i) < w) transposed_matrix[(idy + i) * h + idx] = shared_matrix[threadIdx.x][threadIdx.y + i];
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
