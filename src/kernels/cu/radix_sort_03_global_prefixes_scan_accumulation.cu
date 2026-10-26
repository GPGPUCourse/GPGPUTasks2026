#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* counts,
    const unsigned int* pow2_sums,
          unsigned int* offsets,
    unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const unsigned int len = i;
    unsigned int sum = 0;
    if (len & 1u) {
        sum += counts[len - 1];
    }

    unsigned int offset = 0;
    unsigned int size = (n + 1) / 2;
    for (unsigned int pow2 = 1; pow2 < 32; ++pow2) {
        if (len & (1u << pow2)) {
            sum += pow2_sums[offset + (len >> pow2) - 1];
        }
        offset += size;
        size = (size + 1) / 2;
    }

    offsets[i] = sum;
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &counts, const gpu::gpu_mem_32u &pow2_sums, gpu::gpu_mem_32u &offsets, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(counts.cuptr(), pow2_sums.cuptr(), offsets.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
