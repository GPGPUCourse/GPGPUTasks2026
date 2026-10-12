#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define BLOCK_SIZE 32

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float ac[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bc[BLOCK_SIZE][BLOCK_SIZE];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    unsigned int x = blockIdx.x * BLOCK_SIZE + tx;
    unsigned int y = blockIdx.y * BLOCK_SIZE + ty;

    float cv = 0;

    for(int i = 0; i < (k + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i)
    {
        unsigned int ax = i * BLOCK_SIZE + tx;
        unsigned int ay = y;

        ac[ty][tx] = (ax < k && ay < h) ? a[ay * k + ax] : 0.0f;

        unsigned int bx = x;
        unsigned int by = i * BLOCK_SIZE + ty;

        bc[ty][tx] = (bx < w && by < k) ? b[by * w + bx] : 0.0f;

        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; ++j)
        {
            cv += ac[ty][j] * bc[j][tx];
        }

        __syncthreads();

    }

    if(x < w && y < h)
    {
        c[y * w + x] = cv;
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
