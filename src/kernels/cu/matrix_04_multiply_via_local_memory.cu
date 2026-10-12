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
    const unsigned int col_local = threadIdx.x;
    const unsigned int row_local = threadIdx.y;

    __shared__ float tileA[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float tileB[GROUP_SIZE_Y][GROUP_SIZE_X];
    float acc = 0;

    unsigned int iters = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (unsigned int i = 0; i < iters; i++) {

        unsigned int col_a = i * GROUP_SIZE_X + col_local;
        if (col_a < k && row < h) {
            tileA[row_local][col_local] = a[col_a + row * k];
        } else {
            tileA[row_local][col_local] = 0;
        }

        unsigned int row_b = i * GROUP_SIZE_Y + row_local;
        if (col < w && row_b < k) {
            tileB[row_local][col_local] = b[row_b * w + col];
        } else {
            tileB[row_local][col_local] = 0;
        }

        __syncthreads();

        for (unsigned int j = 0; j < GROUP_SIZE_X; j++) {
            acc += tileA[row_local][j] * tileB[j][col_local];
        }

        __syncthreads();
    }

    if (col < w && row < h) {
        c[col + row * w] = acc;
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
