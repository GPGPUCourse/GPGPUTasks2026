#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

constexpr unsigned int TILE_SIZE = 32;

__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* matrix,            // w x h
    float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h) {

    __shared__ float local_matrix[TILE_SIZE][TILE_SIZE + 1];
    const int base_x = TILE_SIZE * blockIdx.x;
    const int base_y = TILE_SIZE * blockIdx.y;
    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;

    for (int offset = 0; offset < 4; offset += 1) {
        const int x = base_x + local_x;
        const int y = base_y + offset * 8 + local_y;
        if (x < w and y < h) {
            local_matrix[local_y + offset * 8][local_x] = matrix[x + w * y];
        }
    }
    __syncthreads();
    for (int offset = 0; offset < 4; offset += 1) {
        const int x = base_y + local_x;
        const int y = base_x + offset * 8 + local_y;
        if (x < h and y < w) {
            transposed_matrix[x + h * y] = local_matrix[local_x][local_y + offset * 8];
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
        ::matrix_transpose_coalesced_via_local_memory << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (matrix.cuptr(), transposed_matrix.cuptr(), w, h);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
