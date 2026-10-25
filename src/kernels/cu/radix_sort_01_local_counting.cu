#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* input,
          unsigned int* histogram,
    unsigned int n,
    unsigned int bit)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned warp_count = n >> 5;
    const unsigned digit = index < n ? (input[index] >> bit) & (RADIX_BUCKETS - 1) : 0;
#pragma unroll
    for (unsigned bucket = 0; bucket < RADIX_BUCKETS; ++bucket) {
        const unsigned matches = __ballot_sync(0xffffffffu, index < n && digit == bucket);
        if ((threadIdx.x & 31) == 0 && (index >> 5) < warp_count) {
            histogram[bucket * warp_count + (index >> 5)] = __popc(matches);
        }
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
