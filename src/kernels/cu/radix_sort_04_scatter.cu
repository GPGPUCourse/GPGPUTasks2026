#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/radix_sort_common.cu"
#include "../defines.h"

__global__ void __launch_bounds__(GROUP_SIZE) radix_sort_04_scatter(
    const unsigned int* values,
    const unsigned int* block_offsets,
    const unsigned int* digit_starts,
          unsigned int* output,
    unsigned int n,
    unsigned int shift)
{
    constexpr unsigned int WARP_CHUNK = 32 * RADIX_ITEMS_PER_THREAD;

    __shared__ unsigned int s_keys[RADIX_TILE_SIZE];
    __shared__ unsigned int s_warp_hist[NWARPS][RADIX];
    __shared__ unsigned int s_global_shift[RADIX];
    __shared__ unsigned int s_warp_sums[NWARPS];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp = tid / 32;
    const unsigned int lanemask_lt = (1u << lane) - 1;

    #pragma unroll
    for (unsigned int w = 0; w < NWARPS; ++w)
        s_warp_hist[w][tid] = 0;
    const unsigned int global_offset = digit_starts[tid] + block_offsets[tid * gridDim.x + blockIdx.x];

    const unsigned int tile_base = blockIdx.x * RADIX_TILE_SIZE;
    const unsigned int warp_base = tile_base + warp * WARP_CHUNK;

    unsigned int keys[RADIX_ITEMS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < RADIX_ITEMS_PER_THREAD; ++r) {
        const unsigned int i = warp_base + r * 32 + lane;
        keys[r] = i < n ? __ldcs(values + i) : 0xffffffffu;
    }
    __syncthreads();

    unsigned int ranks[RADIX_ITEMS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < RADIX_ITEMS_PER_THREAD; ++r) {
        const unsigned int d = radix_digit(keys[r], shift);
        unsigned int peers = FULL_MASK;
        #pragma unroll
        for (int bit = 0; bit < RADIX_BITS; ++bit) {
            const bool set = (d >> bit) & 1;
            const unsigned int ballot = __ballot_sync(FULL_MASK, set);
            peers &= set ? ballot : ~ballot;
        }
        const unsigned int before = s_warp_hist[warp][d];
        const unsigned int lower_peers = peers & lanemask_lt;
        ranks[r] = before + __popc(lower_peers);
        __syncwarp();
        if (lower_peers == 0)
            s_warp_hist[warp][d] = before + __popc(peers);
        __syncwarp();
    }
    __syncthreads();

    unsigned int digit_count = 0;
    #pragma unroll
    for (unsigned int w = 0; w < NWARPS; ++w) {
        const unsigned int c = s_warp_hist[w][tid];
        s_warp_hist[w][tid] = digit_count;
        digit_count += c;
    }
    unsigned int tile_total;
    const unsigned int tile_start = block_exclusive_scan(digit_count, tile_total, s_warp_sums);
    #pragma unroll
    for (unsigned int w = 0; w < NWARPS; ++w)
        s_warp_hist[w][tid] += tile_start;
    s_global_shift[tid] = global_offset - tile_start;
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < RADIX_ITEMS_PER_THREAD; ++r) {
        const unsigned int d = radix_digit(keys[r], shift);
        s_keys[s_warp_hist[warp][d] + ranks[r]] = keys[r];
    }
    __syncthreads();

    const unsigned int valid = min((unsigned int) RADIX_TILE_SIZE, n - tile_base);
    #pragma unroll
    for (int r = 0; r < RADIX_ITEMS_PER_THREAD; ++r) {
        const unsigned int i = r * GROUP_SIZE + tid;
        if (i < valid) {
            const unsigned int key = s_keys[i];
            output[s_global_shift[radix_digit(key, shift)] + i] = key;
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &block_offsets, const gpu::gpu_mem_32u &digit_starts, gpu::gpu_mem_32u &output, unsigned int n, unsigned int shift)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(block_offsets.size() >= RADIX * workSize.cuGridSize().x, 8543245234123, block_offsets.size());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), block_offsets.cuptr(), digit_starts.cuptr(), output.cuptr(), n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
