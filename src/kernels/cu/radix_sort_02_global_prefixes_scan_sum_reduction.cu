#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort_common.cu"
#include "../defines.h"

#define SCAN_ITEMS_PER_THREAD 4

__global__ void __launch_bounds__(GROUP_SIZE) radix_sort_02_global_prefixes_scan_sum_reduction(
    unsigned int* block_hist,
    unsigned int* digit_totals,
    unsigned int nblocks)
{
    __shared__ unsigned int s_warp_sums[NWARPS];

    const unsigned int tid = threadIdx.x;
    unsigned int* row = block_hist + blockIdx.x * nblocks;

    unsigned int carry = 0;
    for (unsigned int base = 0; base < nblocks; base += GROUP_SIZE * SCAN_ITEMS_PER_THREAD) {
        unsigned int v[SCAN_ITEMS_PER_THREAD];
        unsigned int sum = 0;
        #pragma unroll
        for (int k = 0; k < SCAN_ITEMS_PER_THREAD; ++k) {
            const unsigned int i = base + tid * SCAN_ITEMS_PER_THREAD + k;
            v[k] = i < nblocks ? row[i] : 0;
            sum += v[k];
        }

        unsigned int total;
        unsigned int running = carry + block_exclusive_scan(sum, total, s_warp_sums);
        #pragma unroll
        for (int k = 0; k < SCAN_ITEMS_PER_THREAD; ++k) {
            const unsigned int i = base + tid * SCAN_ITEMS_PER_THREAD + k;
            if (i < nblocks)
                row[i] = running;
            running += v[k];
        }
        carry += total;
    }

    if (tid == 0)
        digit_totals[blockIdx.x] = carry;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize,
    gpu::gpu_mem_32u &block_hist, gpu::gpu_mem_32u &digit_totals, unsigned int nblocks)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(workSize.cuGridSize().x == RADIX, 6543245234123, workSize.cuGridSize().x);
    rassert(block_hist.size() >= RADIX * nblocks, 6543245234124, block_hist.size(), nblocks);
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(block_hist.cuptr(), digit_totals.cuptr(), nblocks);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
