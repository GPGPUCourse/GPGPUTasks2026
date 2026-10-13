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
    const int cache_h = 32;
    const int cache_w = 32;
    curassert(cache_h == cache_w, 2345234523);
    curassert(cache_h == blockDim.y, 1243242);
    curassert(cache_w == blockDim.x, 9112343242);
    __shared__ float cache[cache_h][cache_w];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        cache[threadIdx.y][threadIdx.x] = matrix[w * y + x];
    }
    int ty = blockIdx.x * blockDim.x + threadIdx.y;
    int tx = blockIdx.y * blockDim.y + threadIdx.x;
    __syncthreads();
    if (tx < h && ty < w) {
        transposed_matrix[h * ty + tx] = cache[threadIdx.x][threadIdx.y];
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
