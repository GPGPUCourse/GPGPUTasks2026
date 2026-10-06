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
    __shared__ float tile[GROUP_SIZE_X][GROUP_SIZE_X + 1];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int x = blockIdx.x * GROUP_SIZE_X + tx;
    const unsigned int tileY = blockIdx.y * GROUP_SIZE_X;

    for (unsigned int row = ty;
         row < GROUP_SIZE_X;
         row += GROUP_SIZE_Y)
    {
        const unsigned int y = tileY + row;

        if (x < w && y < h) {
            tile[row][tx] = matrix[size_t(y) * w + x];
        }
    }

    __syncthreads();

    const unsigned int outX = blockIdx.y * GROUP_SIZE_X + tx;
    const unsigned int outTileY = blockIdx.x * GROUP_SIZE_X;

    for (unsigned int row = ty;
         row < GROUP_SIZE_X;
         row += GROUP_SIZE_Y)
    {
        const unsigned int outY = outTileY + row;

        if (outX < h && outY < w) {
            transposed_matrix[size_t(outY) * h + outX] = tile[tx][row];
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
