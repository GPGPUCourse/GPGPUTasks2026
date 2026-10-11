#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>
#include <thread>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* __restrict__ matrix,
    float* __restrict__ transposed_matrix,
    unsigned int w,
    unsigned int h)
{
    constexpr unsigned int TILE = 64;
    constexpr unsigned int ROWS = 16;

    __shared__ float tile[TILE][TILE + 1];

    unsigned int x = blockIdx.x * TILE + threadIdx.x;
    unsigned int y = blockIdx.y * TILE + threadIdx.y;

    for (unsigned int j = 0; j < TILE; j += ROWS) {
        if (x < w && y + j < h) {
            tile[threadIdx.y + j][threadIdx.x] = matrix[(y + j) * w + x];
        }
    }

    __syncthreads();

    unsigned int out_x = blockIdx.y * TILE + threadIdx.x;
    unsigned int out_y = blockIdx.x * TILE + threadIdx.y;

    for (unsigned int j = 0; j < TILE; j += ROWS) {
        if (out_x < h && out_y + j < w) {
            transposed_matrix[(out_y + j) * h + out_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
