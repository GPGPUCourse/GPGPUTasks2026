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

__device__ __forceinline__
void warpHistogramAndRank(
    uint32_t group,
    int lane,
    GroupCounts &hist,
    uint32_t &my_rank)
{
    const uint32_t lower_mask = (1u << lane) - 1;
    GroupCounts h = {};
    uint32_t rank = 0;

    #pragma unroll
    for (int i = 0; i < NGROUPS; ++i) {
        uint32_t b = __ballot_sync(0xFFFFFFFFu, group == (uint32_t)i);
        h[i] = __popc(b);
        if ((uint32_t)i == group) {
            rank = __popc(b & lower_mask);
        }
    }

    if (lane == 31) {
      hist = h;
    }
    my_rank = rank;
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

    uint32_t group_scan = 0;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_idx = threadIdx.x;
    uint32_t warp_idx = threadIdx.x / 32;

    uint32_t elem = (idx < n) ? elems[idx] : -1u;
    uint32_t mask = (1 << ITER_BITS) - 1;
    uint32_t group = (elem >> bit) & mask; 
    uint32_t my_rank = 0;
    warpHistogramAndRank(group, threadIdx.x%32, warp_counts[warp_idx], my_rank);
    __syncthreads();

    for (int wi = 0; wi < warp_idx; ++wi) {
      group_scan += warp_counts[wi][group];
    }
    group_scan += my_rank;

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

    group_scan += tailCounts[group];

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
    res_offset += group_scan;

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
