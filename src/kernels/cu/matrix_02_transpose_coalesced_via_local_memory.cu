#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define TRANSP_TILE 32

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    __shared__ float buffer[TRANSP_TILE][TRANSP_TILE+1];

    const unsigned int x = blockIdx.x * TRANSP_TILE + threadIdx.x;
    const unsigned int y = blockIdx.y * TRANSP_TILE + threadIdx.y;
    if (x < w && y < h) {
        buffer[threadIdx.y][threadIdx.x] = matrix[y * w + x];
    }

    __syncthreads();

    const unsigned int out_x = blockIdx.y * TRANSP_TILE + threadIdx.x;
    const unsigned int out_y = blockIdx.x * TRANSP_TILE + threadIdx.y;
    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = buffer[threadIdx.x][threadIdx.y];
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
