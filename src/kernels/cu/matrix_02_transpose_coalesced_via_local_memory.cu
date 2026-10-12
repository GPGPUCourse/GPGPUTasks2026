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
    __shared__ float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    const unsigned int lx = threadIdx.x;
    const unsigned int ly = threadIdx.y;
    const unsigned int x = blockIdx.x * GROUP_SIZE_X + lx;
    const unsigned int y = blockIdx.y * GROUP_SIZE_Y + ly;
    if (x < w && y < h)
        tile[ly][lx] = matrix[y * w + x];
    __syncthreads();
    const unsigned int out_x = blockIdx.y * GROUP_SIZE_Y + lx;
    const unsigned int out_y = blockIdx.x * GROUP_SIZE_X + ly;
    if (out_x < h && out_y < w)
        transposed_matrix[out_y * h + out_x] = tile[lx][ly];
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
