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
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Во избежание банк-конфликтов делаем биекцию:
    // логический индекс -->  (x, y) <-> ((x + y) % GROUP_SIZE_X, y)  <-- физический индекс
    __shared__ float local_data[GROUP_SIZE_Y][GROUP_SIZE_X];
    if (x < w && y < h) {
        local_data[threadIdx.y][(threadIdx.x + threadIdx.y) % GROUP_SIZE_X] = matrix[y * w + x];
    }
    __syncthreads();

    const auto tx = blockIdx.y * blockDim.y + threadIdx.x;
    const auto ty = blockIdx.x * blockDim.x + threadIdx.y;
    if (tx < h && ty < w) {
        transposed_matrix[ty * h + tx] = local_data[threadIdx.x][(threadIdx.x + threadIdx.y) % GROUP_SIZE_X];
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
