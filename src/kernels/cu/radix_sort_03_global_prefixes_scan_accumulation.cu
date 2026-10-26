#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort_common.cu"
#include "../defines.h"

__global__ void __launch_bounds__(GROUP_SIZE) radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* digit_totals,
          unsigned int* digit_starts)
{
    __shared__ unsigned int s_warp_sums[NWARPS];

    const unsigned int tid = threadIdx.x;
    unsigned int total;
    digit_starts[tid] = block_exclusive_scan(digit_totals[tid], total, s_warp_sums);
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &digit_totals, gpu::gpu_mem_32u &digit_starts)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(workSize.cuGridSize().x == 1 && workSize.cuBlockSize().x == RADIX, 7543245234123, workSize.cuGridSize().x, workSize.cuBlockSize().x);
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(digit_totals.cuptr(), digit_starts.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
