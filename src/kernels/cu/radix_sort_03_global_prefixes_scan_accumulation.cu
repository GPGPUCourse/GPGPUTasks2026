#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <cuda/atomic>

#include "helpers/rassert.cu"
#include "../defines.h"

#define D 8
#define NUM_PACKS 4 // 32 / D
#define NUM_BINS 256 // 2^D
#define ITEMS_PER_THREAD 8
#define WARPS_PER_BLOCK 8
#define THREADS_PER_WARP 32

__shared__ uint32_t warp_histogram[WARPS_PER_BLOCK][NUM_BINS];

__device__ __forceinline__
void scan256(const uint32_t* input, uint32_t* output)
{
    if (threadIdx.x >= 32)
        return;

    uint4 a = *reinterpret_cast<const uint4 *>(input + threadIdx.x * 4);
    uint4 b = *reinterpret_cast<const uint4 *>(input + GROUP_SIZE / 2 + threadIdx.x * 4);

    uint32_t sumA = a.x + a.y + a.z + a.w;
    uint32_t sumB = b.x + b.y + b.z + b.w;

    uint32_t prefixA = sumA;
    uint32_t prefixB = sumB;

#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
        uint32_t prevA = __shfl_up_sync(0xFFFFFFFF, prefixA, offset);
        uint32_t prevB = __shfl_up_sync(0xFFFFFFFF, prefixB, offset);

        if (threadIdx.x >= offset)
        {
            prefixA += prevA;
            prefixB += prevB;
        }
    }

    uint32_t totalA = __shfl_sync(0xFFFFFFFF, prefixA, 31);

    uint32_t offsetA = prefixA - sumA;
    uint32_t offsetB = totalA + prefixB - sumB;

    uint4 resultA = make_uint4(offsetA, offsetA + a.x, offsetA + a.x + a.y, offsetA + a.x + a.y + a.z);
    uint4 resultB = make_uint4(offsetB, offsetB + b.x, offsetB + b.x + b.y, offsetB + b.x + b.y + b.z);

    *reinterpret_cast<uint4 *>(output + threadIdx.x * 4) = resultA;
    *reinterpret_cast<uint4 *>(output + GROUP_SIZE / 2 + threadIdx.x * 4) = resultB;
}

__device__ __forceinline__
uint32_t process_value(const uint32_t pack)
{
    uint32_t groupMask = __match_any_sync(0xFFFFFFFF, pack);

    uint32_t warpId = threadIdx.x / 32;
    uint32_t laneId = threadIdx.x % 32;

    uint32_t total_group_threads = __popc(groupMask); // - сколько потоков в warp'е в нашей группе - на это число нужно увеличить счетчик
    uint32_t lower_threads_mask = (1u << laneId) - 1;
    uint32_t total_before_threads = __popc(groupMask & lower_threads_mask); // - сколько потоков перед нами в нашей группе - это warpRank

    uint32_t lower_group_thread = __ffs(groupMask) - 1;

    uint32_t previousCount = 0;

    if (laneId == lower_group_thread && pack < NUM_BINS)
    {
        previousCount = atomicAdd(
            &warp_histogram[warpId][pack],
            total_group_threads
        );
    }

    previousCount = __shfl_sync(0xFFFFFFFF, previousCount, lower_group_thread);

    return previousCount + total_before_threads;
}

__global__ void radix_sort_03_global_prefixes_scan_accumulation(const uint32_t *input, uint32_t *states, uint32_t *tileCounter, const uint32_t *global_histogram, uint32_t *output, const uint32_t pack_mask, const size_t n)
{
    __shared__ uint32_t tileIdx;

    if (threadIdx.x == 0)
    {
        tileIdx = atomicAdd(tileCounter, 1u);
    }

    __syncthreads();

    constexpr size_t TILE_SIZE = GROUP_SIZE * ITEMS_PER_THREAD;

    size_t baseIdx = static_cast<size_t>(tileIdx) * TILE_SIZE;

    uint32_t warpId = threadIdx.x / 32;
    uint32_t laneId = threadIdx.x % 32;

#pragma unroll
    for (auto idx = laneId; idx < NUM_BINS; idx += THREADS_PER_WARP)
    {
        warp_histogram[warpId][idx] = 0;
    }

    __syncthreads();

    size_t warpBase = baseIdx + warpId * THREADS_PER_WARP * ITEMS_PER_THREAD;

    uint32_t ranks[ITEMS_PER_THREAD];
    uint32_t values[ITEMS_PER_THREAD];

    uint32_t shift = (__ffs(pack_mask) - 1);

#pragma unroll
    for (size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        size_t idx = warpBase + i * THREADS_PER_WARP + laneId;

        uint32_t pack = NUM_BINS;

        if (idx < n)
        {
            values[i] = input[idx];
            pack = (values[i] & pack_mask) >> shift;
        }

        ranks[i] = process_value(pack);
    }

    __shared__ __align__(16) uint32_t block_histogram[NUM_BINS];

    __syncthreads();

    uint32_t pack = threadIdx.x;
    uint32_t prefix = 0;

#pragma unroll
    for (uint32_t warp = 0; warp < WARPS_PER_BLOCK; ++warp)
    {
        uint32_t count = warp_histogram[warp][pack];

        warp_histogram[warp][pack] = prefix;

        prefix += count;
    }

    block_histogram[pack] = prefix;

    __syncthreads();

    __shared__ __align__(16) uint32_t block_offset[NUM_BINS];

    scan256(block_histogram, block_offset);

    __syncthreads();

    uint32_t localIndices[ITEMS_PER_THREAD];

#pragma unroll
    for (size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        size_t idx = warpBase + i * THREADS_PER_WARP + laneId;

        if (idx >= n)
        {
            continue;
        }

        uint32_t value = values[i];
        uint32_t pack_i = (value >> shift) & 0xFF;

        uint32_t blockRank = warp_histogram[warpId][pack_i] + ranks[i];
        localIndices[i] = block_offset[pack_i] + blockRank;
    }

    __syncthreads();


    uint32_t *local_sorted = &warp_histogram[0][0];

#pragma unroll
    for (size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        size_t idx = warpBase + i * THREADS_PER_WARP + laneId;

        if (idx >= n)
        {
            continue;
        }

        local_sorted[localIndices[i]] = values[i];
    }

    __syncthreads();

    // 00|... - N
    // 10|... - L
    // 11|... - G
    constexpr uint32_t STATE_L = 0x8000'0000u;
    constexpr uint32_t STATE_G = 0xC000'0000u;
    constexpr uint32_t STATUS_MASK = STATE_G;
    constexpr uint32_t COUNT_MASK = 0x3FFF'FFFFu;

    const size_t numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    const size_t stateIdx = static_cast<size_t>(tileIdx) * NUM_BINS + pack;

    const uint32_t count = block_histogram[pack];

    __shared__ uint32_t tile_offsets[NUM_BINS];

    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> myState(states[stateIdx]);

    myState.store(STATE_L | count, cuda::memory_order_release);

    uint32_t l_count = 0;
    for (int i = static_cast<int>(tileIdx) - 1; i >= 0; --i)
    {
        const size_t prevIdx = static_cast<size_t>(i) * NUM_BINS + pack;

        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> prevState(states[prevIdx]);

        uint32_t state;

        do
        {
            state = prevState.load(cuda::memory_order_acquire);
        } while ((state & STATUS_MASK) == 0);

        l_count += state & COUNT_MASK;

        if ((state & STATUS_MASK) == STATE_G)
        {
            break;
        }
    }

    tile_offsets[pack] = l_count;
    myState.store(STATE_G | (l_count + count), cuda::memory_order_release);

    __syncthreads();

    const size_t validCount = min(TILE_SIZE, n - baseIdx);

    const uint32_t *global_offsets = global_histogram + (shift / D) * NUM_BINS;

#pragma unroll
    for (size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        size_t localIdx = threadIdx.x + GROUP_SIZE * i;

        if (localIdx >= validCount)
        {
            continue;
        }

        uint32_t value = local_sorted[localIdx];
        uint32_t pack_i = (value >> shift) & 0xFF;

        uint32_t blockRank = localIdx - block_offset[pack_i];

        size_t output_idx = static_cast<size_t>(global_offsets[pack_i]) + tile_offsets[pack_i] + blockRank;

        output[output_idx] = value;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(
    const gpu::WorkSize &workSize,
    const uint32_t *input,
    uint32_t *states,
    uint32_t *tileCounter,
    const uint32_t *global_histogram,
    uint32_t *output,
    const uint32_t pack_mask,
    const size_t n
    )
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input,
        states,
        tileCounter,
        global_histogram,
        output,
        pack_mask,
        n
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
