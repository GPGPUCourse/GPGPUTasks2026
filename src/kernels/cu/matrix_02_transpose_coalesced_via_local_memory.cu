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
    __shared__ float local_mem[32][32];
    const unsigned int from_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int from_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (from_x < w && from_y < h)
        local_mem[threadIdx.y][threadIdx.x ^ threadIdx.y] = matrix[from_y * w + from_x];

    __syncthreads();

    const unsigned int to_x = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int to_y = blockIdx.x * blockDim.x + threadIdx.y;

    if (to_x < h && to_y < w)
        transposed_matrix[to_y * h + to_x] = local_mem[threadIdx.x][threadIdx.y ^ threadIdx.x];
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
