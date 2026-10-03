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
    __shared__ float tile[32][33];
    unsigned int x = blockIdx.x * 32u + threadIdx.x;
    unsigned int y = blockIdx.y * 32u + threadIdx.y;
    if (x < w && y < h) {
        tile[threadIdx.y][threadIdx.x] = matrix[(size_t)y * w + x];
    }
    __syncthreads();
    unsigned int col = blockIdx.x * 32u + threadIdx.y;
    unsigned int row = blockIdx.y * 32u + threadIdx.x;
    if (col < w && row < h) {
        transposed_matrix[(size_t)col * h + row] = tile[threadIdx.x][threadIdx.y];
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
