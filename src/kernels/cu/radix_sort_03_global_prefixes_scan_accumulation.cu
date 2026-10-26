#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* level_sums,
    unsigned int* offsets,
    unsigned int n,
    unsigned int level)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (level == 0)
        offsets[i] = 0;
    unsigned int segment_size = 1u << level;
    if ((i & segment_size) != 0) {
        unsigned int segment = (i >> level) - 1;
        offsets[i] += level_sums[segment];
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1, gpu::gpu_mem_32u& buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
