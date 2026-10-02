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
    __shared__ float alocal[MATMUL_DIM][MATMUL_DIM];
    __shared__ float blocal[MATMUL_DIM][MATMUL_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t localX = threadIdx.x;
    uint32_t localY = threadIdx.y;
    float res = 0.0f;

    for (int bi = 0; bi < k/MATMUL_DIM; ++bi) {
      if (x < w && y < h) {
        alocal[localY][localX] = a[y * k + bi * MATMUL_DIM + localX];
        blocal[localY][localX] = b[(bi * MATMUL_DIM + localY) * w + x];
      }
      __syncthreads();
      if (x < w && y < h) {
        for (int j = 0; j < MATMUL_DIM; ++j) {
          res += alocal[localY][j] * blocal[j][localX];
        }
      }
      __syncthreads();
    }

    if (x < w && y < h) {
      c[y * w + x] = res;
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
