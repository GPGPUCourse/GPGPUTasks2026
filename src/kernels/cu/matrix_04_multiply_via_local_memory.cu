#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define HD __host__ __device__

struct float2x2 {
  HD float2x2(float2 row1, float2 row2) {
    col[0] = { row1.x, row2.x };
    col[1] = { row1.y, row2.y };
  }
  float2x2() = default;
  float2 col[2];
};

HD void fma2x2(const float2x2 &a, const float2x2 &b, float2x2 &acc) {
  acc.col[0].x += a.col[0].x * b.col[0].x;
  acc.col[0].x += a.col[1].x * b.col[0].y;
  acc.col[0].y += a.col[0].y * b.col[0].x;
  acc.col[0].y += a.col[1].y * b.col[0].y;
  acc.col[1].x += a.col[0].x * b.col[1].x;
  acc.col[1].x += a.col[1].x * b.col[1].y;
  acc.col[1].y += a.col[0].y * b.col[1].x;
  acc.col[1].y += a.col[1].y * b.col[1].y;
}

__global__ void matrix_multiply_via_local_memory(
                       const float2* a, // rows=h x cols=k
                       const float2* b, // rows=k x cols=w
                             float2* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float2x2 alocal[MATMUL_DIM][MATMUL_DIM];
    __shared__ float2x2 blocal[MATMUL_DIM][MATMUL_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t localX = threadIdx.x;
    uint32_t localY = threadIdx.y;
    float2x2 res = {};

    uint32_t w2 = w/2;
    uint32_t h2 = h/2;
    uint32_t k2 = k/2;

    for (int bi = 0; bi < k2/MATMUL_DIM; ++bi) {
      if (x < w2 && y < h2) {
        alocal[localY][localX] = { a[y*2 * k2 + bi * MATMUL_DIM + localX], a[(y*2+1) * k2 + bi * MATMUL_DIM + localX] };
        blocal[localY][localX] = { b[(bi * MATMUL_DIM + localY) * 2 * w2 + x], b[((bi * MATMUL_DIM + localY)*2 + 1) * w2 + x] };
      }
      __syncthreads();
      if (x < w2 && y < h2) {
        for (int j = 0; j < MATMUL_DIM; ++j) {
          fma2x2(alocal[localY][j], blocal[j][localX], res);
        }
      }
      __syncthreads();
    }

    if (x < w2 && y < h2) {
      c[y * 2 * w2 + x] = { res.col[0].x, res.col[1].x };
      c[(y*2+1) * w2 + x] = { res.col[0].y, res.col[1].y }; 
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>((float2*)a.cuptr(), (float2*)b.cuptr(), (float2*)c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
