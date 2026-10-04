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
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int local_i = threadIdx.x;
    const unsigned int local_j = threadIdx.y;

    __shared__ float blockA[MULT_BLOCK_SIZE][MULT_BLOCK_SIZE];
    __shared__ float blockB[MULT_BLOCK_SIZE][MULT_BLOCK_SIZE];

    float acc = 0;
    for (unsigned int block = 0; block < (k + MULT_BLOCK_SIZE - 1) / MULT_BLOCK_SIZE; ++block) {
        unsigned int a_col = block * MULT_BLOCK_SIZE + threadIdx.x;
        unsigned int b_row = block * MULT_BLOCK_SIZE + threadIdx.y;

        if (j < h && a_col < k) {
            blockA[local_j][local_i] = a[k * j + a_col];
        } else {
            blockA[local_j][local_i] = 0.0;
        }

        if (i < w && b_row < k) {
            blockB[local_j][local_i] = b[w * b_row + i];
        } else {
            blockB[local_j][local_i] = 0.0;
        }
        __syncthreads();

        for (unsigned int it = 0; it < MULT_BLOCK_SIZE; ++it) {
            acc += blockA[local_j][it] * blockB[it][local_i];
        }
        __syncthreads();
    }

    if (i < w && j < h) {
        c[j * w + i] = acc;
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
