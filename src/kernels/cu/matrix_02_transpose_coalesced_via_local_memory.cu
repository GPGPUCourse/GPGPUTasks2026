#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define BLOCK_SIZE 32

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    __shared__ float lm[BLOCK_SIZE][BLOCK_SIZE];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    unsigned int x = blockIdx.x * BLOCK_SIZE + tx;
    unsigned int y = blockIdx.y * BLOCK_SIZE + ty;

    if(x < w && y < h)
    {
        lm[ty][(tx + ty) % BLOCK_SIZE] = matrix[y * w + x];
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + tx;
    y = blockIdx.x * BLOCK_SIZE + ty;

    if(x < h && y < w)
    {
        transposed_matrix[y * h + x] = lm[tx][(tx + ty) % BLOCK_SIZE];
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
