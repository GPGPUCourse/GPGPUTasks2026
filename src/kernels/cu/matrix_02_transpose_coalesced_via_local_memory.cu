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
    constexpr unsigned int tile_size = 32;
    __shared__ float tile[tile_size][tile_size + 1];
    const unsigned int x = blockIdx.x * tile_size + threadIdx.x;
    const unsigned int y = blockIdx.y * tile_size + threadIdx.y;
    for (unsigned int offset = 0; offset < tile_size; offset += 8)
        if (x < w && y + offset < h)
            tile[threadIdx.y + offset][threadIdx.x] = matrix[size_t(y + offset) * w + x];
    __syncthreads();

    const unsigned int out_x = blockIdx.y * tile_size + threadIdx.x;
    const unsigned int out_y = blockIdx.x * tile_size + threadIdx.y;
    for (unsigned int offset = 0; offset < tile_size; offset += 8)
        if (out_x < h && out_y + offset < w)
            transposed_matrix[size_t(out_y + offset) * h + out_x] = tile[threadIdx.x][threadIdx.y + offset];
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
