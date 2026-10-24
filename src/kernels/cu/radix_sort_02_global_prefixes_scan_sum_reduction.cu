#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    uint2* counts,
    unsigned int n,
    unsigned int offset)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint32_t zeros, ones;
    uint2 cur_counts = (idx < n) ? counts[offset + idx] : uint2{0, 0};
    if (threadIdx.x == 0) {
      zeros = 0;
      ones = 0;
    }
    __syncthreads();
    atomicAdd(&zeros, cur_counts.x);
    atomicAdd(&ones, cur_counts.y);
    __syncthreads();
    if (threadIdx.x == 0) {
      counts[offset + n + blockIdx.x] = { zeros, ones };
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(
    const gpu::WorkSize &workSize, gpu::gpu_mem_32u &counts, unsigned int n,
    unsigned int offset) {
  gpu::Context context;
  rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312,
          context.type());
  cudaStream_t stream = context.cudaStream();
  ::radix_sort_02_global_prefixes_scan_sum_reduction<<<
      workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
      (uint2 *)counts.cuptr(), n, offset);
  CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
