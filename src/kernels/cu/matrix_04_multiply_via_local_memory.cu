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
    const uint X = blockDim.x * blockIdx.x + threadIdx.x;
    const uint Y = blockDim.y * blockIdx.y + threadIdx.y;
    const uint tx = threadIdx.x;
    const uint ty = threadIdx.y;

    __shared__ float mini_a[16 * 16];
    __shared__ float mini_b[16 * 16];

    float sum = 0.0f;
    for (int i = 0; i < k / 16; i++) {
        mini_a[ty * 16 + tx] = a[Y * k + i * 16 + tx];
        mini_b[ty * 16 + tx] = b[(i * 16 + ty) * w + X];
        __syncthreads();
        for (int m = 0; m < 16; m++) {
            sum += mini_a[ty * 16 + m] * mini_b[m * 16 + tx];
        }
        __syncthreads();
    }
    c[Y * w + X] = sum;
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
