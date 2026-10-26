#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort_common.cu"
#include "../defines.h"

__global__ void __launch_bounds__(GROUP_SIZE) radix_sort_01_local_counting(
    const unsigned int* values,
          unsigned int* block_hist,
    unsigned int n,
    unsigned int shift)
{
    __shared__ unsigned int s_hist[NWARPS][RADIX];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / 32;

    #pragma unroll
    for (unsigned int w = 0; w < NWARPS; ++w)
        s_hist[w][tid] = 0;
    __syncthreads();

    const unsigned int tile_base = blockIdx.x * RADIX_TILE_SIZE;
    #pragma unroll
    for (int r = 0; r < RADIX_ITEMS_PER_THREAD / 4; ++r) {
        const unsigned int i = tile_base + (r * GROUP_SIZE + tid) * 4;
        if (i + 4 <= n) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(values + i));
            atomicAdd(&s_hist[warp][radix_digit(v.x, shift)], 1);
            atomicAdd(&s_hist[warp][radix_digit(v.y, shift)], 1);
            atomicAdd(&s_hist[warp][radix_digit(v.z, shift)], 1);
            atomicAdd(&s_hist[warp][radix_digit(v.w, shift)], 1);
        } else {
            for (unsigned int k = i; k < n && k < i + 4; ++k)
                atomicAdd(&s_hist[warp][radix_digit(values[k], shift)], 1);
        }
    }
    __syncthreads();

    unsigned int sum = 0;
    #pragma unroll
    for (unsigned int w = 0; w < NWARPS; ++w)
        sum += s_hist[w][tid];
    block_hist[tid * gridDim.x + blockIdx.x] = sum;
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &block_hist, unsigned int n, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(block_hist.size() >= RADIX * workSize.cuGridSize().x, 5462345234123, block_hist.size());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), block_hist.cuptr(), n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
