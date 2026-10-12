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
    __shared__ float as[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float bs[GROUP_SIZE_Y][GROUP_SIZE_X];
    const unsigned int lx = threadIdx.x;
    const unsigned int ly = threadIdx.y;
    const unsigned int col = blockIdx.x * GROUP_SIZE_X + lx;
    const unsigned int row = blockIdx.y * GROUP_SIZE_Y + ly;
    float sum = 0.0f;
    const unsigned int tiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
    for (unsigned int t = 0; t < tiles; ++t) {
        const unsigned int a_col = t * GROUP_SIZE_X + lx;
        const unsigned int b_row = t * GROUP_SIZE_X + ly;
        as[ly][lx] = (row < h && a_col < k) ? a[row * k + a_col] : 0.0f;
        bs[ly][lx] = (b_row < k && col < w) ? b[b_row * w + col] : 0.0f;
        __syncthreads();
        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i)
            sum += as[ly][i] * bs[i][lx];
        __syncthreads();
    }
    if (row < h && col < w)
        c[row * w + col] = sum;
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
