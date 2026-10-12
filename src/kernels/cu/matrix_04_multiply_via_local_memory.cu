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
    constexpr unsigned int tile_size = 16;

    const unsigned int tile_x = threadIdx.x;
    const unsigned int tile_y = threadIdx.y;
    const unsigned int x = blockIdx.x * blockDim.x + tile_x;
    const unsigned int y = blockIdx.y * blockDim.y + tile_y;

    __shared__ float a_tile[tile_size][tile_size];
    __shared__ float b_tile[tile_size][tile_size];

    float c_sum = 0.0f;

    for (unsigned int tile_start = 0; tile_start < k; tile_start += tile_size) {
        const unsigned int a_col = tile_start + tile_x;
        const unsigned int b_row = tile_start + tile_y;

        a_tile[tile_y][tile_x] = (y < h && a_col < k) ? a[y * k + a_col] : 0.0f;
        b_tile[tile_y][tile_x] = (b_row < k && x < w) ? b[b_row * w + x] : 0.0f;

        __syncthreads();

        for (unsigned int t = 0; t < tile_size; ++t) {
            c_sum += a_tile[tile_y][t] * b_tile[t][tile_x];
        }

        __syncthreads();
    }

    if (x < w && y < h) {
        c[y * w + x] = c_sum;
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
