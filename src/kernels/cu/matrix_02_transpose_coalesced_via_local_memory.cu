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
    unsigned int x = blockIdx.x * BANK_SIZE + threadIdx.x;

    __shared__ float local_matrix[BANK_SIZE][BANK_SIZE + 1];

    // printf("y: %d, x: %d\n", threadIdx.y, threadIdx.x);
    #pragma unroll
    for (unsigned int step = 0; step < BANK_SIZE; step += 8) {
        unsigned int y = blockIdx.y * BANK_SIZE + threadIdx.y + step;
        if (x < w && y < h) {
            local_matrix[threadIdx.y + step][threadIdx.x] = matrix[y * w + x];
        }
    }

    __syncthreads();

    unsigned int new_x = blockIdx.y * BANK_SIZE + threadIdx.x;

    #pragma unroll
    for (int step = 0; step < BANK_SIZE; step += 8) {
        unsigned int new_y = blockIdx.x * BANK_SIZE + threadIdx.y + step;

        if (new_x < h && new_y < w) {
            transposed_matrix[new_y * h + new_x] = local_matrix[threadIdx.x][threadIdx.y + step];
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
