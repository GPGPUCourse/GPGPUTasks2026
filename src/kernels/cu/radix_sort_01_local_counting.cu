#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* keys,
          unsigned int* counters,
    unsigned int n,
    unsigned int bit_shift,
    unsigned int n_groups)
{
    __shared__ unsigned int hist[RADIX];

    const unsigned int lid = threadIdx.x;
    const unsigned int group = blockIdx.x;
    const unsigned int base = group * SORT_TILE;

    for (unsigned int b = lid; b < RADIX; b += SORT_WG_SIZE) {
        hist[b] = 0;
    }
    __syncthreads();

    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        const unsigned int i = base + lid + k * SORT_WG_SIZE;
        if (i < n) {
            atomicAdd(&hist[(keys[i] >> bit_shift) & RADIX_MASK], 1u);
        }
    }
    __syncthreads();

    for (unsigned int b = lid; b < RADIX; b += SORT_WG_SIZE) {
        counters[b * n_groups + group] = hist[b];
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &keys, gpu::gpu_mem_32u &counters, unsigned int n, unsigned int bit_shift, unsigned int n_groups)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(keys.cuptr(), counters.cuptr(), n, bit_shift, n_groups);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
