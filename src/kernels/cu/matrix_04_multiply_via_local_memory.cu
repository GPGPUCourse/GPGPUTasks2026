#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define MUL_TILE 16

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float buffer_a[MUL_TILE][MUL_TILE];
    __shared__ float buffer_b[MUL_TILE][MUL_TILE];

    const unsigned int col = blockIdx.x * MUL_TILE + threadIdx.x;
    const unsigned int row = blockIdx.y * MUL_TILE + threadIdx.y;

    float acc = 0.0f;

    for (unsigned int t = 0; t < k; t += MUL_TILE) {
        const unsigned int a_col = t + threadIdx.x;
        const unsigned int b_row = t + threadIdx.y;

        if (row < h && a_col < k) 
            buffer_a[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        else 
            buffer_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < k && col < w) 
            buffer_b[threadIdx.y][threadIdx.x] = b[b_row * w + col];
        else 
            buffer_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (unsigned int kk = 0; kk < MUL_TILE; ++kk) {
            acc += buffer_a[threadIdx.y][kk] * buffer_b[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < h && col < w) {
        c[row * w + col] = acc;
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
