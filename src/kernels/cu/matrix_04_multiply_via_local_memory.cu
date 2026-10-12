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
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    __shared__ float tile1[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float tile2[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    for (unsigned int t = 0; t < (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X; ++t) {
        const unsigned int col1 = t * GROUP_SIZE_X + tx;
        const unsigned int row2 = t * GROUP_SIZE_Y + ty;
        tile1[ty][tx] = (y < h && col1 < k) ? a[y * k + col1] : 0.0f;
        tile2[ty][tx] = (row2 < k && x < w) ? b[row2 * w + x] : 0.0f;

        __syncthreads();

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i)
            sum += tile1[ty][i] * tile2[i][tx];

        __syncthreads();
    }
    if (x < w && y < h)
        c[y * w + x] = sum;
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
