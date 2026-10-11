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
    __shared__ float a_local[BANK_SIZE + 1][BANK_SIZE];
    __shared__ float b_local[BANK_SIZE + 1][BANK_SIZE];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int num_phases = (k + BANK_SIZE - 1) / BANK_SIZE;

    float acc = 0.0f;

    for (int phase = 0; phase < num_phases; ++phase) {
        unsigned int a_col = phase * BANK_SIZE + tx;
        if (x < w) {
            a_local[ty][tx] = a[y * k + a_col];
        } else {
            a_local[ty][tx] = 0.0f;
        }

        unsigned int b_row = phase * BANK_SIZE + ty;
        if (y < h) {
            b_local[ty][tx] = b[b_row * w + x];
        } else {
            b_local[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (unsigned int i = 0; i < BANK_SIZE; ++i) {
            acc += a_local[ty][i] * b_local[i][tx];
        }

        __syncthreads();
    }

    if (y < h && x < w) {
        c[y * w + x] = acc;
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
