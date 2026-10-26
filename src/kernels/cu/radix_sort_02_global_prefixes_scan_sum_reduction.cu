#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
          unsigned int* data,
          unsigned int* block_sums,
    unsigned int n)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    __shared__ unsigned int local_counters[GROUP_SIZE];

    local_counters[tid] = id < n ? data[id] : 0;
    __syncthreads();

    #pragma unroll
    for (unsigned int offset = 1; offset < GROUP_SIZE; offset *= 2) {
        unsigned int add = tid >= offset ? local_counters[tid - offset] : 0;
        __syncthreads();
        local_counters[tid] += add;
        __syncthreads();
    }

    if (id < n) {
        data[id] = local_counters[tid];
    }
    if (tid == GROUP_SIZE - 1) {
        block_sums[blockIdx.x] = local_counters[tid];
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
      unsigned int* data,
      unsigned int* block_sums,
unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data, block_sums, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
