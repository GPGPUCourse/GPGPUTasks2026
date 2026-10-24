#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

struct GroupCounts {
  __host__ __device__ uint32_t &operator[](uint32_t i) { return data[i]; }
  __host__ __device__ uint32_t operator[](uint32_t i) const { return data[i]; }
  uint32_t data[NGROUPS];
};

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // TODO try char
    GroupCounts* counts,
    unsigned int n,
    unsigned int offset)
{
    __shared__ GroupCounts groups_cnt;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    GroupCounts elem = (idx < n) ? counts[offset + idx] : GroupCounts{};
    if (threadIdx.x == 0) {
      for (int i = 0; i < (1 << ITER_BITS); ++i) {
        groups_cnt[i] = 0;
      }
    }
    __syncthreads();
    for (uint32_t i = 0; i < NGROUPS; ++i) {
      atomicAdd(groups_cnt.data + i, elem[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      counts[offset + n + blockIdx.x] = groups_cnt;
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
      (GroupCounts*)counts.cuptr(), n, offset);
  CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
