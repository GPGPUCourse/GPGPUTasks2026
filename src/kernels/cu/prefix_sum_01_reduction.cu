#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include "../defines.h"
#include "helpers/rassert.cu"
#include <cuda/atomic>

using PrefixState = unsigned long long;
using PrefixAtomic = cuda::atomic_ref<PrefixState, cuda::thread_scope_device>;

__device__ __forceinline__ uint4 prefix_sum_load_four(
    const unsigned int* input,
    unsigned int i,
    unsigned int n)
{
    if (i < n && n - i >= 4)
        return *reinterpret_cast<const uint4*>(input + i);

    return make_uint4(
        i < n ? input[i] : 0u,
        i + 1 < n ? input[i + 1] : 0u,
        i + 2 < n ? input[i + 2] : 0u,
        i + 3 < n ? input[i + 3] : 0u);
}

__device__ __forceinline__ unsigned int prefix_sum_warp_scan(
    unsigned int value,
    unsigned int lane)
{
    for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        unsigned int left = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset)
            value += left;
    }
    return value;
}

__device__ __forceinline__ unsigned int prefix_sum_look_back(
    PrefixState* states,
    unsigned int tile,
    unsigned int lane)
{
    if (tile == 0)
        return 0;

    int position = static_cast<int>(tile) - 1;
    unsigned int prefix = 0;

    while (true) {
        int index = position - static_cast<int>(lane);
        PrefixState state;
        unsigned int ready_prefix;
        unsigned int last;

        while (true) {
            state = index >= 0
                ? PrefixAtomic(states[index]).load(cuda::memory_order_relaxed)
                : (PrefixState(2) << 32);

            unsigned int status = static_cast<unsigned int>(state >> 32);
            ready_prefix = __ballot_sync(0xffffffffu, status == 2);
            unsigned int empty = __ballot_sync(0xffffffffu, status == 0);
            last = ready_prefix != 0 ? __ffs(ready_prefix) - 1 : 31u;
            unsigned int needed = last == 31
                ? 0xffffffffu : (1u << (last + 1)) - 1u;

            if ((empty & needed) == 0)
                break;
        }

        unsigned int value = lane <= last
            ? static_cast<unsigned int>(state) : 0u;

        for (unsigned int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);

        prefix += __shfl_sync(0xffffffffu, value, 0);

        if (ready_prefix != 0)
            return prefix;

        position -= 32;
    }
}

template<unsigned int BLOCK>
__device__ __forceinline__ void prefix_sum_block_sync()
{
    if (BLOCK == 32)
        __syncwarp(0xffffffffu);
    else
        __syncthreads();
}

template<unsigned int BLOCK, unsigned int TILE>
__global__ void prefix_sum_lookback_kernel(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    PrefixState* states,
    unsigned int* next_tile,
    unsigned int n)
{
    static_assert(BLOCK == 256 || BLOCK == 128 || BLOCK == 64 || BLOCK == 32,
        "unsupported block size");
    static_assert(TILE == 1024 || TILE == 2048 || TILE == 4096,
        "unsupported tile size");
    constexpr unsigned int WARPS = BLOCK / 32;
    constexpr unsigned int STRIPS = TILE / (BLOCK * 4);
    constexpr unsigned int SEGMENTS = TILE / 128;

    __shared__ unsigned int tile;
    __shared__ unsigned int warp_prefix[SEGMENTS];
    __shared__ unsigned int tile_prefix;

    unsigned int tx = threadIdx.x;
    unsigned int lane = tx & 31u;
    unsigned int warp = tx >> 5;

    if (tx == 0)
        tile = atomicAdd(next_tile, 1u);

    prefix_sum_block_sync<BLOCK>();

    uint4 data[STRIPS];
    unsigned int value[STRIPS];

#pragma unroll
    for (unsigned int strip = 0; strip < STRIPS; ++strip) {
        unsigned int i = tile * TILE + strip * BLOCK * 4u + 4u * tx;
        data[strip] = prefix_sum_load_four(input, i, n);
        data[strip].y += data[strip].x;
        data[strip].z += data[strip].y;
        data[strip].w += data[strip].z;

        value[strip] = prefix_sum_warp_scan(data[strip].w, lane);

        if (lane == 31)
            warp_prefix[strip * WARPS + warp] = value[strip];
    }

    prefix_sum_block_sync<BLOCK>();

    if (warp == 0) {
        unsigned int total = lane < SEGMENTS ? warp_prefix[lane] : 0u;
        total = prefix_sum_warp_scan(total, lane);

        if (lane < SEGMENTS)
            warp_prefix[lane] = total;

        unsigned int tile_sum = __shfl_sync(0xffffffffu, total, SEGMENTS - 1);

        if (lane == 0) {
            PrefixState status = tile == 0 ? 2 : 1;
            PrefixAtomic(states[tile]).store(
                (status << 32) | PrefixState(tile_sum),
                cuda::memory_order_relaxed);
        }

        unsigned int prefix = prefix_sum_look_back(states, tile, lane);

        if (lane == 0) {
            tile_prefix = prefix;
            if (tile != 0) {
                unsigned int inclusive = prefix + tile_sum;
                PrefixAtomic(states[tile]).store(
                    (PrefixState(2) << 32) | PrefixState(inclusive),
                    cuda::memory_order_relaxed);
            }
        }
    }

    prefix_sum_block_sync<BLOCK>();

#pragma unroll
    for (unsigned int strip = 0; strip < STRIPS; ++strip) {
        unsigned int segment = strip * WARPS + warp;
        unsigned int offset = value[strip] - data[strip].w + tile_prefix;
        if (segment > 0)
            offset += warp_prefix[segment - 1];

        data[strip].x += offset;
        data[strip].y += offset;
        data[strip].z += offset;
        data[strip].w += offset;

        unsigned int i = tile * TILE + strip * BLOCK * 4u + 4u * tx;
        if (i < n && n - i >= 4) {
            *reinterpret_cast<uint4*>(output + i) = data[strip];
        } else {
            if (i < n)
                output[i] = data[strip].x;
            if (i + 1 < n)
                output[i + 1] = data[strip].y;
            if (i + 2 < n)
                output[i + 2] = data[strip].z;
            if (i + 3 < n)
                output[i + 3] = data[strip].w;
        }
    }
}

namespace cuda {

template<unsigned int BLOCK>
void prefix_sum_launch_lookback(
    const gpu::WorkSize& workSize,
    const unsigned int* input,
    unsigned int* output,
    PrefixState* states,
    unsigned int* next_tile,
    unsigned int n,
    unsigned int tile_size,
    cudaStream_t stream)
{
    switch (tile_size) {
    case 1024:
        ::prefix_sum_lookback_kernel<BLOCK, 1024>
            <<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
                input, output, states, next_tile, n);
        break;
    case 2048:
        ::prefix_sum_lookback_kernel<BLOCK, 2048>
            <<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
                input, output, states, next_tile, n);
        break;
    case 4096:
        ::prefix_sum_lookback_kernel<BLOCK, 4096>
            <<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
                input, output, states, next_tile, n);
        break;
    }
}

void prefix_sum_lookback(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input,
    gpu::gpu_mem_32u& output,
    gpu::gpu_mem_32u& workspace,
    unsigned int n,
    unsigned int tile_size)
{
    if (n == 0)
        return;

    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA,
        34523543124312, context.type());
    rassert(tile_size == 1024 || tile_size == 2048 || tile_size == 4096,
        7104202613, tile_size);

    unsigned int tiles = n / tile_size + (n % tile_size != 0);
    size_t workspace_size = size_t(tiles) * 2 + 1;
    rassert(workspace.number() >= workspace_size,
        7104202610, workspace.number(), workspace_size);
    rassert((workSize.cuBlockSize().x == 256
                || workSize.cuBlockSize().x == 128
                || workSize.cuBlockSize().x == 64
                || workSize.cuBlockSize().x == 32)
            && workSize.cuBlockSize().y == 1
            && workSize.cuBlockSize().z == 1,
        7104202611, workSize.cuBlockSize().x);
    rassert(workSize.cuGridSize().x == tiles
            && workSize.cuGridSize().y == 1
            && workSize.cuGridSize().z == 1,
        7104202612, workSize.cuGridSize().x, tiles);

    cudaStream_t stream = context.cudaStream();
    CUDA_SAFE_CALL(cudaMemsetAsync(workspace.cuptr(), 0,
        workspace_size * sizeof(unsigned int), stream));

    PrefixState* states = reinterpret_cast<PrefixState*>(workspace.cuptr());
    unsigned int* next_tile = workspace.cuptr() + size_t(tiles) * 2;

    switch (workSize.cuBlockSize().x) {
    case 256:
        prefix_sum_launch_lookback<256>(workSize, input.cuptr(), output.cuptr(),
            states, next_tile, n, tile_size, stream);
        break;
    case 128:
        prefix_sum_launch_lookback<128>(workSize, input.cuptr(), output.cuptr(),
            states, next_tile, n, tile_size, stream);
        break;
    case 64:
        prefix_sum_launch_lookback<64>(workSize, input.cuptr(), output.cuptr(),
            states, next_tile, n, tile_size, stream);
        break;
    case 32:
        prefix_sum_launch_lookback<32>(workSize, input.cuptr(), output.cuptr(),
            states, next_tile, n, tile_size, stream);
        break;
    }

    CUDA_CHECK_KERNEL_SYNC(stream);
}

}
