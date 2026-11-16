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

    if (x >= w || y >= h) {
        return;
    }

    float acc = 0.0f;

    // Rectangular support in release 1.0
    const unsigned int p = GROUP_SIZE_X;
    __shared__ float local_data_a[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __shared__ float local_data_b[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    for (int i = 0; i < k; i += p) {
        __syncthreads();

        if (i + tx < k) {
            local_data_a[ty][tx] = a[y * k + i + tx];
        }

        if (i + ty < k) {
            local_data_b[ty][tx] = b[(i + ty) * w + x];
        }

        __syncthreads();

        for (int j = 0; j < p; ++j) {
            if (i + j < k) {
                acc += local_data_a[ty][j] * local_data_b[j][tx];
            }
        }
    }

    c[y * w + x] = acc; 
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
