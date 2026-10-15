#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ void warpScanAndReduce(uint32_t value, uint32_t& scan, uint32_t& total) {
  const unsigned mask = 0xffffffff;
  const uint32_t lane = threadIdx.x & 31;

  scan = value;
  #pragma unroll
  for (uint32_t offset = 1; offset < 32; offset <<= 1) {
      uint32_t n = __shfl_up_sync(mask, scan, offset);
      if (lane >= offset) scan += n;
  }
  total = __shfl_sync(mask, scan, 31);
}

__global__ void prefix_sum_02_prefix_accumulation(
    const uint32_t *elems,
    const unsigned int* pow2_sum,
    unsigned int* prefix_sum_accum,
    unsigned int n)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t elems_count = idx + 1;
  uint32_t local_idx = threadIdx.x;
  uint32_t warp_idx = local_idx / 32;
  
  __shared__ uint32_t warp_sum[GROUP_SIZE/32];
  uint32_t res = 0;

  uint32_t elem = (idx < n) ? elems[idx] : 0;
  uint32_t warp_prefix_sum = 0;
  uint32_t cur_warp_sum = 0;
  warpScanAndReduce(elem, warp_prefix_sum, cur_warp_sum);
  if (local_idx % 32 == 0) {
    warp_sum[warp_idx] = cur_warp_sum;
  }
  __syncthreads();

  res += warp_prefix_sum;
  for (int wi = 0; wi < warp_idx; ++wi) {
    res += warp_sum[wi];
  }

  uint32_t offset = 0;
  uint32_t cur_aligned = idx >> 1;
  uint32_t pow_elems_count = (n+1)/2;
  for (int i = 0; i < 7; ++i) {
    cur_aligned >>= 1;
    offset += pow_elems_count;
    pow_elems_count = (pow_elems_count + 1) / 2; 
  }
  
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
