#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define HD __host__ __device__

struct float4x4 {
  HD float4x4(float4 row0, float4 row1, float4 row2, float4 row3) {
    f[0]  = row0.x; f[1]  = row1.x; f[2]  = row2.x; f[3]  = row3.x;
    f[4]  = row0.y; f[5]  = row1.y; f[6]  = row2.y; f[7]  = row3.y;
    f[8]  = row0.z; f[9]  = row1.z; f[10] = row2.z; f[11] = row3.z;
    f[12] = row0.w; f[13] = row1.w; f[14] = row2.w; f[15] = row3.w;
  }
  float4x4() = default;
  float f[16];
};

HD void fma2x2(const float4x4 &a, const float4x4 &b, float4x4 &acc) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float sum = 0.0f;
      #pragma unroll
      for (int t = 0; t < 4; ++t) {
        sum += a.f[t * 4 + i] * b.f[j * 4 + t];
      }
      acc.f[j * 4 + i] += sum;
    }
  }
}

__global__ void matrix_multiply_via_local_memory(
                       const float4* a,
                       const float4* b,
                             float4* c,
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float4x4 alocal[MATMUL_DIM][MATMUL_DIM];
    __shared__ float4x4 blocal[MATMUL_DIM][MATMUL_DIM];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t localX = threadIdx.x;
    uint32_t localY = threadIdx.y;
    float4x4 res = {};

    uint32_t w4 = w/4;
    uint32_t h4 = h/4;
    uint32_t k4 = k/4;

    for (int bi = 0; bi < k4/MATMUL_DIM; ++bi) {
      if (x < w4 && y < h4) {
        alocal[localY][localX] = float4x4{
          a[(y*4+0) * k4 + bi * MATMUL_DIM + localX],
          a[(y*4+1) * k4 + bi * MATMUL_DIM + localX],
          a[(y*4+2) * k4 + bi * MATMUL_DIM + localX],
          a[(y*4+3) * k4 + bi * MATMUL_DIM + localX]
        };
        blocal[localY][localX] = float4x4{
          b[(bi * MATMUL_DIM + localY) * 4 * w4 + x],
          b[((bi * MATMUL_DIM + localY)*4 + 1) * w4 + x],
          b[((bi * MATMUL_DIM + localY)*4 + 2) * w4 + x],
          b[((bi * MATMUL_DIM + localY)*4 + 3) * w4 + x]
        };
      }
      __syncthreads();
      if (x < w4 && y < h4) {
        for (int j = 0; j < MATMUL_DIM; ++j) {
          fma2x2(alocal[localY][j], blocal[j][localX], res);
        }
      }
      __syncthreads();
    }

    if (x < w4 && y < h4) {
      c[(y*4+0) * w4 + x] = { res.f[0], res.f[4], res.f[8],  res.f[12] };
      c[(y*4+1) * w4 + x] = { res.f[1], res.f[5], res.f[9],  res.f[13] };
      c[(y*4+2) * w4 + x] = { res.f[2], res.f[6], res.f[10], res.f[14] };
      c[(y*4+3) * w4 + x] = { res.f[3], res.f[7], res.f[11], res.f[15] };
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>((float4*)a.cuptr(), (float4*)b.cuptr(), (float4*)c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda