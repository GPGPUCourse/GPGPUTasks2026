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
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int tj = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int ti = blockIdx.x * blockDim.x + threadIdx.y;

    const unsigned int idx = j + i * w;
    const unsigned int tidx = tj + ti * h;

    const unsigned int local_idx = threadIdx.x + threadIdx.y * (blockDim.x + 1);
    const unsigned int tlocal_idx = threadIdx.y + threadIdx.x * (blockDim.x + 1);

    __shared__ float local_data[(GROUP_SIZE_X + 1) * GROUP_SIZE_Y];

    if (j < w && i < h)
        local_data[local_idx] = matrix[idx];

    __syncthreads();

    if (tj < h && ti < w)
        transposed_matrix[tidx] = local_data[tlocal_idx];
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
