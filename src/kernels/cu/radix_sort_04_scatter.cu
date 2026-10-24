#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

struct GroupCounts {
  __host__ __device__ uint32_t &operator[](uint32_t i) { return data[i]; }
  __host__ __device__ uint32_t operator[](uint32_t i) const { return data[i]; }
  __host__ __device__ GroupCounts& operator+=(const GroupCounts &rhs) {
    for (int i = 0; i < NGROUPS; ++i) {
      data[i] += rhs.data[i];
    }
    return *this;
  }
  uint32_t data[NGROUPS];
};

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
    const GroupCounts* counts,
    unsigned int *result,
    unsigned int bit,
    unsigned int n)
{
    __shared__ GroupCounts warp_counts[GROUP_SIZE/32];
    __shared__ GroupCounts tailCounts;

    GroupCounts warp_scan = {};
    GroupCounts scan = {};

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_idx = threadIdx.x;
    uint32_t warp_idx = threadIdx.x / 32;

    uint32_t elem = (idx < n) ? elems[idx] : -1u;
    uint32_t mask = (1 << ITER_BITS) - 1;
    uint32_t group = (elem >> bit) & mask; 
    for (int i = 0; i < NGROUPS; ++i) {
      warp_scan[i] = warpScan(group == i);
    }
    if (threadIdx.x % 32 == 31) {
      warp_counts[threadIdx.x/32] = warp_scan;
    }
    __syncthreads();

    scan += warp_scan;
    for (int wi = 0; wi < warp_idx; ++wi) {
      scan += warp_counts[wi];
    }

    uint32_t offset = 0;
    uint32_t count = blockIdx.x;
    uint32_t pow_count = (n+255) / 256;

    GroupCounts sum = {};
    while(count > 0) {
      uint32_t mod = count % 256;

      GroupCounts my_value = local_idx < mod ? counts[offset + (count - mod) + local_idx] : GroupCounts{};
      sum += my_value;
      offset += pow_count;
      pow_count = (pow_count + 255) / 256;
      count /= 256;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      tailCounts = GroupCounts{};
    }
    __syncthreads();
    for (int i = 0; i < NGROUPS; ++i) {
      atomicAdd(tailCounts.data + i, sum[i]);
    }
    __syncthreads();

    scan += tailCounts;

    GroupCounts totalCnt = {};
    {
      uint32_t offset = 0;
      uint32_t pow_count = (n+255) / 256;
      while(pow_count > 1) {
        offset += pow_count;
        pow_count = (pow_count + 255) / 256;
      }
      totalCnt = counts[offset];
    }

    uint32_t res_offset = 0;
    for (int i = 0; i < group; ++i) {
      res_offset += totalCnt[i];
    }
    res_offset += scan[group]-1;

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
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), (GroupCounts*)counts.cuptr(), output.cuptr(), bit, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
