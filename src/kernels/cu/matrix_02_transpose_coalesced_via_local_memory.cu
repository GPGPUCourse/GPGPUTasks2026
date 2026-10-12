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
    const unsigned int TILE = 32;
    const unsigned int BLOCK_ROWS = 8;

    __shared__ float tile[TILE][TILE + 1];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const unsigned int i = blockIdx.x * TILE + tx;
    const unsigned int j0 = blockIdx.y * TILE + ty;
    for (unsigned int r = 0; r < TILE; r += BLOCK_ROWS) {
        if (i < w && j0 + r < h)
            tile[ty + r][tx] = matrix[(j0 + r) * w + i];
    }

    __syncthreads();

    const unsigned int out_i = blockIdx.y * TILE + tx;
    const unsigned int out_j0 = blockIdx.x * TILE + ty;
    for (unsigned int r = 0; r < TILE; r += BLOCK_ROWS) {
        if (out_i < h && out_j0 + r < w)
            transposed_matrix[(out_j0 + r) * h + out_i] = tile[tx][ty + r];
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
