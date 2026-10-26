#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

namespace {
constexpr unsigned int BLOCK_SIZE = 256u;
constexpr unsigned int TILE_WARP_SIZE = 32u;
constexpr unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / TILE_WARP_SIZE;
constexpr unsigned int RADIX_BITS = 4u;
constexpr unsigned int NUM_BINS = 1u << RADIX_BITS;
constexpr unsigned int BIN_MASK = NUM_BINS - 1u;
constexpr unsigned int INVALID_BIN = 0xFFFFFFFFu;
constexpr unsigned int FULL_MASK = 0xFFFFFFFFu;
} // namespace

__global__ void radix_sort_04_scatter(
    const unsigned int* in_array,
    const unsigned int* scanned_offsets,
          unsigned int* out_array,
    unsigned int n,
    unsigned int shift)
{
    __shared__ unsigned int s_warp_counts[NUM_BINS][WARPS_PER_BLOCK];
    __shared__ unsigned int s_bin_start[NUM_BINS];
    __shared__ unsigned int s_global_base[NUM_BINS];
    __shared__ unsigned int s_sorted_vals[BLOCK_SIZE];
    __shared__ unsigned int s_sorted_bins[BLOCK_SIZE];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (TILE_WARP_SIZE - 1u);
    const unsigned int warp_id = tid / TILE_WARP_SIZE;
    const unsigned int idx = blockIdx.x * BLOCK_SIZE + tid;
    const unsigned int valid_in_block = (idx - tid < n) ? min(BLOCK_SIZE, n - (idx - tid)) : 0u;

    const unsigned int val = (idx < n) ? in_array[idx] : 0u;
    const unsigned int my_bin = (idx < n) ? ((val >> shift) & BIN_MASK) : INVALID_BIN;

    // 1. Считаем ранг элемента внутри своего варпа через битовую маску __ballot_sync
    unsigned int warp_rank = 0u;
    const unsigned int lower_lanes_mask = (1u << lane) - 1u;

    #pragma unroll
    for (unsigned int b = 0u; b < NUM_BINS; ++b) {
        const unsigned int match_mask = __ballot_sync(FULL_MASK, my_bin == b);
        if (lane == 0u) {
            s_warp_counts[b][warp_id] = __popc(match_mask);
        }
        if (my_bin == b) {
            warp_rank = __popc(match_mask & lower_lanes_mask);
        }
    }
    __syncthreads();

    // 2. Первые 16 потоков считают префиксы варпов для своей корзины
    if (tid < NUM_BINS) {
        unsigned int sum = 0u;
        for (unsigned int w = 0u; w < WARPS_PER_BLOCK; ++w) {
            const unsigned int count = s_warp_counts[tid][w];
            s_warp_counts[tid][w] = sum;
            sum += count;
        }
        s_bin_start[tid] = sum; // Пока храним общее число элементов корзины tid в блоке
        s_global_base[tid] = scanned_offsets[tid * gridDim.x + blockIdx.x];
    }
    __syncthreads();

    // 3. Поток 0 превращает размеры корзин блока в стартовые позиции в shared memory
    if (tid == 0u) {
        unsigned int offset = 0u;
        for (unsigned int b = 0u; b < NUM_BINS; ++b) {
            const unsigned int count = s_bin_start[b];
            s_bin_start[b] = offset;
            offset += count;
        }
    }
    __syncthreads();

    // 4. Упорядочиваем элементы локально в shared memory
    if (idx < n) {
        const unsigned int local_pos = s_bin_start[my_bin] + s_warp_counts[my_bin][warp_id] + warp_rank;
        s_sorted_vals[local_pos] = val;
        s_sorted_bins[local_pos] = my_bin;
    }
    __syncthreads();

    // 5. Связная (coalesced) запись в глобальную память
    if (tid < valid_in_block) {
        const unsigned int b = s_sorted_bins[tid];
        const unsigned int rank_in_bin = tid - s_bin_start[b];
        out_array[s_global_base[b] + rank_in_bin] = s_sorted_vals[tid];
    }
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
