#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    const unsigned int* pow2_sum, 
    unsigned int* next_pow2_sum,
    unsigned int n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint32_t sum;
    if (threadIdx.x == 0) {
      sum = 0;
    }
    uint32_t my_value = (idx < n) ? pow2_sum[idx] : 0;
    __syncthreads();
    atomicAdd(&sum, my_value);
    __syncthreads();
    if (threadIdx.x == 0) {
      next_pow2_sum[blockIdx.x] = sum;
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize, const uint32_t *in,
                             uint32_t *out, unsigned int n) {
  gpu::Context context;
  rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312,
          context.type());
  cudaStream_t stream = context.cudaStream();
  ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0,
                              stream>>>(in, out, n);
  CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
