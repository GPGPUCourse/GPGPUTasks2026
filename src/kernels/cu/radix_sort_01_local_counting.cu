#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* buffer1,
          unsigned int* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    const unsigned int n = a1;
    const unsigned int shift = a2;
    const unsigned int nGroups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    __shared__ unsigned int hist[NBINS];

    const unsigned int i = threadIdx.x;
    const unsigned int lane = i % 32;
    const unsigned int index = blockIdx.x * GROUP_SIZE + i;

    if (i < NBINS)
        hist[i] = 0;
    __syncthreads();

    const bool valid = index < n;
    const unsigned int d = valid ? ((buffer1[index] >> shift) & (NBINS - 1)) : 0;

    unsigned int mask = __ballot_sync(0xFFFFFFFFu, valid);
    for (int b = 0; b < RADIX_BITS; ++b) {
        const bool bit = (d >> b) & 1u;
        const unsigned int bal = __ballot_sync(0xFFFFFFFFu, bit);
        mask &= bit ? bal : ~bal;
    }

    if (valid && (mask & ((1u << lane) - 1u)) == 0)
        atomicAdd(&hist[d], __popc(mask));
    __syncthreads();

    if (i < NBINS)
        buffer2[i * nGroups + blockIdx.x] = hist[i];
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
