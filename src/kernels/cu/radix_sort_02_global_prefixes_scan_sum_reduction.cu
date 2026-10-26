#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

namespace {
constexpr unsigned int SCAN_BLOCK_SIZE = 256u;
} // namespace

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* counters,
          unsigned int* part_sums,
    unsigned int total_counters)
{
    __shared__ unsigned int s_sum[SCAN_BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;
    const unsigned int num_parts = gridDim.x;
    const unsigned int part_size = (total_counters + num_parts - 1u) / num_parts;
    const unsigned int start = blockIdx.x * part_size;
    const unsigned int end = min(start + part_size, total_counters);

    unsigned int local_sum = 0u;
    for (unsigned int i = start + tid; i < end; i += SCAN_BLOCK_SIZE) {
        local_sum += counters[i];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (unsigned int stride = SCAN_BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        part_sums[blockIdx.x] = s_sum[0];
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
