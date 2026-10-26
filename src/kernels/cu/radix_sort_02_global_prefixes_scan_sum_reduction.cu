#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    const unsigned int* data,
          unsigned int* block_sums,
    unsigned int n)
{
    __shared__ unsigned int scratch[SCAN_WG_SIZE];

    const unsigned int lid = threadIdx.x;
    const unsigned int base = blockIdx.x * SCAN_TILE;

    unsigned int sum = 0;
    for (unsigned int k = 0; k < SCAN_IPT; ++k) {
        const unsigned int i = base + lid + k * SCAN_WG_SIZE;
        if (i < n) {
            sum += data[i];
        }
    }

    scratch[lid] = sum;
    __syncthreads();

    for (unsigned int stride = SCAN_WG_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        __syncthreads();
    }

    if (lid == 0) {
        block_sums[blockIdx.x] = scratch[0];
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &data, gpu::gpu_mem_32u &block_sums, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data.cuptr(), block_sums.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
