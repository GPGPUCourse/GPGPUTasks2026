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
    const int TILE = 32;
    curassert(TILE == blockDim.y, 1243242);
    curassert(TILE == blockDim.x, 9112343242);
    __shared__ float A[TILE][TILE];
    __shared__ float B[TILE][TILE];

    int col = blockDim.x * blockIdx.x;
    int row = blockDim.y * blockIdx.y;

    float sum = 0.0f;
    for (int i = 0; TILE * i < k + TILE - 1; i++) {
        int a_row = row + threadIdx.y;
        int a_col = threadIdx.x + TILE * i;
        if (a_row < h && a_col < k) {
            A[threadIdx.y][threadIdx.x] = a[a_row * k + a_col];
        } else {
            A[threadIdx.y][threadIdx.x] = 0;
        }

        int b_row = threadIdx.y + TILE * i;
        int b_col = col + threadIdx.x;
        if (b_row < k && b_col < w) {
            B[threadIdx.y][threadIdx.x] = b[b_row * w + b_col];
        } else {
            B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();
        for (int j = 0; j < TILE; j++) {
            // мы потоками одного варпа обращаемся к разным банкам бродкастом. возможно быстрее в один банк запихать
            sum += A[threadIdx.y][j] * B[j][threadIdx.x];
        }

        __syncthreads();
    }
    if (row + threadIdx.y < h && col + threadIdx.x < w) {
        c[(row + threadIdx.y) * w + col + threadIdx.x] = sum;
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
