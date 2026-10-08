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
    const unsigned int index = y * w + x;
    if (x >= w || y >= h) {
        return;
    }

    __shared__ float buff_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float buff_b[GROUP_SIZE_Y][GROUP_SIZE_X];
    float sum = 0;
    for (unsigned i = 0; i < k; i += GROUP_SIZE_Y) {
        buff_a[threadIdx.y][threadIdx.x] = a[y * k + i + threadIdx.x];
        buff_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * w + x];

        __syncthreads();
        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j) {
            sum += buff_a[threadIdx.y][j] * buff_b[j][threadIdx.x];
        }
        __syncthreads();
    }

    c[index] = sum;
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
