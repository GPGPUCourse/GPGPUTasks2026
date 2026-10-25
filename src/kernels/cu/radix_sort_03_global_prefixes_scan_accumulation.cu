#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

//task04: prefix_sum_02_prefix_accumulation
__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* pow2_sums,
    const unsigned int* histogram,
          unsigned int* bucket_offsets,
    unsigned int n)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp_base = index & ~31;
    unsigned big_prefix_sum = 0;
    for (unsigned level = 5, acc = 0; level < 32 && (n >> level) != 0; ++level) {
        if (index < n && (warp_base >> level & 1)) {
            big_prefix_sum += pow2_sums[acc + (warp_base >> level) - 1];
        }
        acc += n >> level;
    }

    const unsigned value = index < n ? histogram[index] : 0;
    unsigned small_prefix_sum = value;
#pragma unroll
    for (unsigned offset = 1; offset < 32; offset <<= 1) {
        const unsigned preceding = __shfl_up_sync(0xffffffffu, small_prefix_sum, offset);
        if (lane >= offset) {
            small_prefix_sum += preceding;
        }
    }
    if (index < n) {
        bucket_offsets[index] = big_prefix_sum + small_prefix_sum - value;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
