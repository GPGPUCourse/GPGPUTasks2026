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
    // TODO
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[32][32 + 1];
    for (int i = 0; i < 4; ++i) {
        if (idx < w && idy * 4 + i < h) {
            tile[threadIdx.y * 4 + i][threadIdx.x] = matrix[(idy * 4 + i) * w + idx];
        }
    }

    __syncthreads();

    int to_idx = blockIdx.y * blockDim.y * 4 + threadIdx.x;
    int to_idy = blockIdx.x * blockDim.x;
    for (int i = 0; i < 4; ++i) {
        if (to_idx < h &&to_idy + 4 * threadIdx.y + i < w) {
            transposed_matrix[(to_idy + 4 * threadIdx.y + i) * h + to_idx] = tile[threadIdx.x][threadIdx.y * 4 + i];
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
