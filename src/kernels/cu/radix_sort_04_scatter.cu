#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* buffer1,
    const unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    const unsigned int n = a1;
    const unsigned int shift = a2;
    const unsigned int nGroups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    constexpr unsigned int NWARPS = GROUP_SIZE / 32;

    __shared__ unsigned int warpHist[NWARPS][NBINS];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp = tid / 32;
    const unsigned int i = blockIdx.x * GROUP_SIZE + tid;

    for (unsigned int j = tid; j < NWARPS * NBINS; j += GROUP_SIZE)
        (&warpHist[0][0])[j] = 0;
    __syncthreads();

    const bool valid = i < n;
    const unsigned int x = valid ? buffer1[i] : 0;
    const unsigned int d = (x >> shift) & (NBINS - 1);

    unsigned int mask = __ballot_sync(0xFFFFFFFFu, valid);
    for (int b = 0; b < RADIX_BITS; ++b) {
        const bool bit = (d >> b) & 1u;
        const unsigned int bal = __ballot_sync(0xFFFFFFFFu, bit);
        mask &= bit ? bal : ~bal;
    }

    const unsigned int rankInWarp = __popc(mask & ((1u << lane) - 1u));
    if (valid && rankInWarp == 0)
        warpHist[warp][d] = __popc(mask);
    __syncthreads();

    if (!valid)
        return;

    unsigned int warpOffset = 0;
    for (unsigned int w = 0; w < warp; ++w)
        warpOffset += warpHist[w][d];

    const unsigned int idx = d * nGroups + blockIdx.x;
    const unsigned int base = (idx == 0) ? 0 : buffer2[idx - 1];

    buffer3[base + warpOffset + rankInWarp] = x;
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
