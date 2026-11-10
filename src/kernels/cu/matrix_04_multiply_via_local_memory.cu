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
    // TODO
    __shared__ float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];
    const unsigned int x = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    const unsigned int y = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;

    float sum = 0.0f;
    for (unsigned int start = 0; start < k; start += GROUP_SIZE_X) {
        const unsigned int a_col = start + threadIdx.x;
        const unsigned int b_row = start + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = y < h && a_col < k ? a[size_t(y) * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = b_row < k && x < w ? b[size_t(b_row) * w + x] : 0.0f;
        __syncthreads();

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i)
            sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        __syncthreads();
    }
    if (x < w && y < h)
        c[size_t(y) * w + x] = sum;
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
