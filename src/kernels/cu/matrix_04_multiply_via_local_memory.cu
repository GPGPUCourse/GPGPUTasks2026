#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float local_a[TILE_SIZE][TILE_SIZE];
    __shared__ float local_b[TILE_SIZE][TILE_SIZE];

    const unsigned int i = blockIdx.x * TILE_SIZE + threadIdx.x;
    const unsigned int j = blockIdx.y * TILE_SIZE + threadIdx.y;

    const unsigned int tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;
    float res = 0.f;

    for(int r = 0; r < tile_count; ++r) {
        const unsigned int col_a = r * TILE_SIZE + threadIdx.x;
        if(col_a < k && j < h) {
            local_a[threadIdx.y][threadIdx.x] = a[j * k + col_a];
        }
        else {
            local_a[threadIdx.y][threadIdx.x] = 0.f;
        }

        const unsigned int row_b = r * TILE_SIZE + threadIdx.y;
        if(row_b < k && i < w) {
            local_b[threadIdx.y][threadIdx.x] = b[row_b * w + i];
        }
        else {
            local_b[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        for(int d = 0; d < TILE_SIZE; ++d) {
            res += local_a[threadIdx.y][d] * local_b[d][threadIdx.x];
        }

        __syncthreads();
    }

    if (j < h && i < w) {
        c[j * w + i] = res;
    }

}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
