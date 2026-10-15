#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const uint32_t *elems,
    const unsigned int* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  
  uint32_t elems_count = idx + 1;
  uint32_t res = 0;
  if (elems_count % 2 != 0) {
    res += elems[idx];
  }

  uint32_t offset = 0;
  uint32_t cur_aligned = elems_count >> 1;
  uint32_t pow_elems_count = (n+1)/2;
  while(cur_aligned) {
    if (cur_aligned & 1) {
      res += pow2_sum[offset + cur_aligned - 1];
    }
    cur_aligned >>= 1;
    offset += pow_elems_count;
    pow_elems_count = (pow_elems_count + 1) / 2; 
  }

  prefix_sum_accum[idx] = res;
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
                                       const gpu::gpu_mem_32u &elems,
                                       const gpu::gpu_mem_32u &pow2_sum,
                                       gpu::gpu_mem_32u &prefix_sum_accum,
                                       unsigned int n) {
  gpu::Context context;
  rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312,
          context.type());
  cudaStream_t stream = context.cudaStream();
  ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(),
                                        workSize.cuBlockSize(), 0, stream>>>(
      elems.cuptr(), pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n);
  CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
