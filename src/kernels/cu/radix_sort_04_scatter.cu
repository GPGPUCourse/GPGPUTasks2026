#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ __forceinline__ uint32_t warpScan(uint32_t val) {
    unsigned mask = 0xFFFFFFFF;
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }
    return val;
}

__global__ void radix_sort_04_scatter(
    const unsigned int* elems,
    const uint2* counts,
    unsigned int *result,
    unsigned int bit,
    unsigned int n)
{
    __shared__ uint32_t warp_zeros[GROUP_SIZE/32], warp_ones[GROUP_SIZE/32];
    __shared__ uint32_t tailSum0, tailSum1;

    uint32_t warp_scan0 = 0; 
    uint32_t warp_scan1 = 0;
    uint32_t scan0 = 0;
    uint32_t scan1 = 0;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_idx = threadIdx.x;
    uint32_t warp_idx = threadIdx.x / 32;

    uint32_t elem = (idx < n) ? elems[idx] : -1u;
    uint32_t bitval = (elem >> bit) & 1; 
    warp_scan0 = warpScan(bitval == 0);
    warp_scan1 = warpScan(bitval == 1);
    if (threadIdx.x % 32 == 31) {
      warp_zeros[threadIdx.x/32] = warp_scan0;
      warp_ones[threadIdx.x/32] = warp_scan1;
    }
    __syncthreads();

    scan0 += warp_scan0;
    scan1 += warp_scan1;
    for (int wi = 0; wi < warp_idx; ++wi) {
      scan0 += warp_zeros[wi];
      scan1 += warp_ones[wi];
    }

    uint32_t offset = 0;
    uint32_t count = blockIdx.x;
    uint32_t pow_count = (n+255) / 256;

    uint2 sum = { 0, 0 };
    while(count > 0) {
      uint32_t mod = count % 256;

      uint2 my_value = local_idx < mod ? counts[offset + (count - mod) + local_idx] : uint2{0, 0};
      sum.x += my_value.x;
      sum.y += my_value.y;
      offset += pow_count;
      pow_count = (pow_count + 255) / 256;
      count /= 256;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      tailSum0 = 0;
      tailSum1 = 0;
    }
    __syncthreads();
    atomicAdd(&tailSum0, sum.x);
    atomicAdd(&tailSum1, sum.y);
    __syncthreads();

    scan0 += tailSum0;
    scan1 += tailSum1;

    uint32_t totalCnt0 = 0;
    {
      uint32_t offset = 0;
      uint32_t pow_count = (n+255) / 256;
      while(pow_count > 1) {
        offset += pow_count;
        pow_count = (pow_count + 255) / 256;
      }
      totalCnt0 = counts[offset].x;
    }

    uint32_t res_offset = 0;
    if (bitval == 0) {
      res_offset = scan0 - 1;
    } else {
      res_offset = totalCnt0 + scan1 - 1;
    }

    if (res_offset < n) {
      result[res_offset] = elem;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &input, const gpu::gpu_mem_32u &counts, gpu::gpu_mem_32u &output, unsigned int bit, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), (uint2*)counts.cuptr(), output.cuptr(), bit, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
