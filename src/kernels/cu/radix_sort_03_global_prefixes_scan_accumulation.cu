#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

//task04: prefix_sum_02_prefix_accumulation
__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* pow2_zero_sums,
    const unsigned int* input,
          unsigned int* zero_prefixes,
    unsigned int n,
    unsigned int bit)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const bool is_zero = index < n && (input[index] >> bit & 1) == 0;
    const unsigned zeros = __ballot_sync(0xffffffffu, is_zero);
    if (index >= n) {
        return;
    }

    const unsigned lane = threadIdx.x & 31;
    const unsigned warp_base = index & ~31;
    unsigned big_prefix_sum = 0;
#pragma unroll
    for (unsigned level = 5, acc = 0; level < LOG_N; ++level) {
        if (warp_base >> level & 1) {
            big_prefix_sum += pow2_zero_sums[acc + (warp_base >> level) - 1];
        }
        acc += n >> level;
    }

    const unsigned lanes_before_mask = (1 << lane) - 1;
    const unsigned prefix = big_prefix_sum + __popc(zeros & lanes_before_mask);
    zero_prefixes[index] = prefix;
    if (index == n - 1) {
        zero_prefixes[n] = prefix + is_zero;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
