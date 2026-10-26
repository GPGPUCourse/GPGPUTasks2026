#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* values,
    const unsigned int* offsets,
          unsigned int* sorted,
    unsigned int n,
    unsigned int shift)
{
    __shared__ unsigned int warp_counts[GROUP_SIZE / 32][RADIX_SIZE];

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;

    if (threadIdx.x < GROUP_SIZE / 32 * RADIX_SIZE) {
        warp_counts[threadIdx.x / RADIX_SIZE][threadIdx.x % RADIX_SIZE] = 0;
    }
    __syncthreads();

    unsigned int value = 0;
    unsigned int bucket = RADIX_SIZE;
    if (i < n) {
        value = values[i];
        bucket = (value >> shift) & (RADIX_SIZE - 1);
    }

    const unsigned int same = __match_any_sync(0xffffffff, bucket);
    const unsigned int warp_offset = __popc(same & ((1u << lane) - 1));
    if (warp_offset == 0 && bucket < RADIX_SIZE) {
        warp_counts[warp][bucket] = __popc(same);
    }
    __syncthreads();

    if (i >= n) {
        return;
    }

    unsigned int local_offset = warp_offset;
    for (unsigned int w = 0; w < warp; ++w) {
        local_offset += warp_counts[w][bucket];
    }

    sorted[offsets[bucket * gridDim.x + blockIdx.x] + local_offset] = value;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &offsets, gpu::gpu_mem_32u &sorted, unsigned int n, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), offsets.cuptr(), sorted.cuptr(), n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
