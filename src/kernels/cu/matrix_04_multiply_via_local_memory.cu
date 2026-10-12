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
    __shared__ float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    unsigned int col = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    unsigned int row = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;

    float sum = 0.0f;
    for (unsigned int tile_k = 0; tile_k < k; tile_k += GROUP_SIZE_X) {
        if (row < h && (tile_k + threadIdx.x) < k) {
            a_tile[threadIdx.y][threadIdx.x] = a[row * k + tile_k + threadIdx.x];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((tile_k + threadIdx.y) < k && col < w) {
            b_tile[threadIdx.y][threadIdx.x] = b[(tile_k + threadIdx.y) * w + col];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int ki = 0; ki < GROUP_SIZE_X; ++ki) {
            sum += a_tile[threadIdx.y][ki] * b_tile[ki][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < h && col < w) {
        c[row * w + col] = sum;
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
