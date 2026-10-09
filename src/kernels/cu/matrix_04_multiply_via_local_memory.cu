#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int local_x = threadIdx.x;
    const unsigned int local_y = threadIdx.y;

    __shared__ float local_a[1024];
    __shared__ float local_b[1024];
    float accum = 0;
    for (int k1 = 0; k1 < k; k1 += 32) {
        if (j < h && k1 + local_x < k) {
            local_a[32 * local_y + local_x] = a[j * k + (k1 + local_x)];
        } else {
            local_a[32 * local_y + local_x] = 0;
        }
        if (i < w && k1 + local_y < k) {
            local_b[32 * local_y + local_x] = b[(k1 + local_y) * w + i];
        } else {
            local_b[32 * local_y + local_x] = 0;
        }

        __syncthreads();

        for (int g = 0; g < 32; g++) {
            accum += local_a[32 * local_y + g] * local_b[32 * g + local_x];
        }

        __syncthreads();
    }
    int destination = j * w + i;
    c[destination] = accum;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
