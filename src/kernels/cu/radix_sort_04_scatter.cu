#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* bucket_offsets,
          unsigned int* output,
    unsigned int n,
    unsigned int bit)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned value = index < n ? input[index] : 0;
    const unsigned digit = (value >> bit) & (RADIX_BUCKETS - 1);
    unsigned same_digit_lanes = 0;
#pragma unroll
    for (unsigned bucket = 0; bucket < RADIX_BUCKETS; ++bucket) {
        const unsigned matches = __ballot_sync(0xffffffffu, index < n && digit == bucket);
        if (digit == bucket) {
            same_digit_lanes = matches;
        }
    }

    if (index >= n) {
        return;
    }
    const unsigned lane = threadIdx.x & 31;
    const unsigned local_rank = __popc(same_digit_lanes & (1 << lane) - 1);
    const unsigned warp_count = n >> 5;
    const unsigned dest = bucket_offsets[digit * warp_count + (index >> 5)] + local_rank;
    output[dest] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
