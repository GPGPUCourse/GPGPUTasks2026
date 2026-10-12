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
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float a_tmp[GROUP_SIZE_X * GROUP_SIZE_Y];
    __shared__ float b_tmp[GROUP_SIZE_X * GROUP_SIZE_Y];
    float accum = 0;
    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
        unsigned int a_col = i + threadIdx.x;
        if (a_col < k && y < h) {
            a_tmp[threadIdx.y * GROUP_SIZE_X + threadIdx.x] = a[y * k + a_col];
        }else {
            a_tmp[threadIdx.y * GROUP_SIZE_X + threadIdx.x] = 0;
        }
        unsigned int b_row = i + threadIdx.y;
        if (b_row < k && x < w) {
            b_tmp[threadIdx.y * GROUP_SIZE_X + threadIdx.x] = b[b_row * w + x];
        }else {
            b_tmp[threadIdx.y * GROUP_SIZE_X + threadIdx.x] = 0;
        }
        __syncthreads();

        for (int j = 0; j < GROUP_SIZE_X; j++) {
            accum += a_tmp[threadIdx.y * GROUP_SIZE_X + j] * b_tmp[j * GROUP_SIZE_X + threadIdx.x];
        }
        __syncthreads();
    }
    if (x < w && y < h) {
        c[y * w + x] = accum;
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
