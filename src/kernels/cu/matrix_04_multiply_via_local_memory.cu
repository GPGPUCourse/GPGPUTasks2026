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
    size_t i = threadIdx.y + blockIdx.y * blockDim.y;
    size_t j = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_a[GROUP_SIZE_X * GROUP_SIZE_Y];
    __shared__ float shared_b[GROUP_SIZE_X * GROUP_SIZE_Y];

    float res = 0;
    for (uint step = 0; step < k / GROUP_SIZE_X; ++step) {
        uint shift = step * GROUP_SIZE_X;

        shared_a[threadIdx.y * blockDim.x + threadIdx.x] = a[i * k + threadIdx.x + shift];
        shared_b[threadIdx.y * blockDim.x + threadIdx.x] = b[(threadIdx.y + shift) * w + j];

        __syncthreads();
        for (uint l = 0; l < GROUP_SIZE_X; ++l) {
            res += shared_a[threadIdx.y * blockDim.x + l] * shared_b[l * blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    c[i * w + j] = res;
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
