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
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float local_a[GROUP_SIZE_Y][GROUP_SIZE_X], local_b[GROUP_SIZE_Y][GROUP_SIZE_X];


    float sm = 0;
    for (int i = 0; i < k; i += GROUP_SIZE_X) {
        if (row < h && i + threadIdx.x < k)
            local_a[threadIdx.y][threadIdx.x] = a[row * k + i + threadIdx.x];
        else
            local_a[threadIdx.y][threadIdx.x] = 0;

        if (col < w && i + threadIdx.y < k)
            local_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * w + col];
        else
            local_b[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int j = 0; j < GROUP_SIZE_X; j++) {
            sm += local_a[threadIdx.y][j] * local_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    if (col < w && row < h)
        c[row * w + col] = sm;
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
