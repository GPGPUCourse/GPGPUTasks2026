#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, // contains n values
          unsigned int* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (n+1)/2) {
      return;
    }

    uint32_t next_value = pow2_sum[2 * idx];
    if (2*idx+1 < n) {
      next_value += pow2_sum[2 * idx + 1];
    }

    next_pow2_sum[idx] = next_value;
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
