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

__global__ void prefix_sum_lookback_kernel(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    PrefixState* states,
    unsigned int* next_tile,
    unsigned int n)
{
    __shared__ unsigned int tile;
    __shared__ unsigned int warp_prefix[8];
    __shared__ unsigned int tile_prefix;

    unsigned int tx = threadIdx.x;
    unsigned int lane = tx & 31u;
    unsigned int warp = tx >> 5;

    if (tx == 0)
        tile = atomicAdd(next_tile, 1u);

    __syncthreads();

    unsigned int i = tile * 1024u + 4u * tx;
    uint4 data = prefix_sum_load_four(input, i, n);
    data.y += data.x;
    data.z += data.y;
    data.w += data.z;

    unsigned int thread_sum = data.w;
    unsigned int value = prefix_sum_warp_scan(thread_sum, lane);

    if (lane == 31)
        warp_prefix[warp] = value;

    __syncthreads();

    if (warp == 0) {
        unsigned int total = lane < 8 ? warp_prefix[lane] : 0u;
        total = prefix_sum_warp_scan(total, lane);

        if (lane < 8)
            warp_prefix[lane] = total;

        unsigned int tile_sum = __shfl_sync(0xffffffffu, total, 7);

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

    __syncthreads();

    if (warp > 0)
        value += warp_prefix[warp - 1];

    unsigned int offset = value - thread_sum + tile_prefix;
    data.x += offset;
    data.y += offset;
    data.z += offset;
    data.w += offset;

    if (i < n && n - i >= 4) {
        *reinterpret_cast<uint4*>(output + i) = data;
    } else {
        if (i < n)
            output[i] = data.x;
        if (i + 1 < n)
            output[i + 1] = data.y;
        if (i + 2 < n)
            output[i + 2] = data.z;
        if (i + 3 < n)
            output[i + 3] = data.w;
    }
}

namespace cuda {

void prefix_sum_lookback(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input,
    gpu::gpu_mem_32u& output,
    gpu::gpu_mem_32u& workspace,
    unsigned int n)
{
    if (n == 0)
        return;

    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA,
        34523543124312, context.type());

    unsigned int tiles = n / 1024u + (n % 1024u != 0);
    size_t workspace_size = size_t(tiles) * 2 + 1;
    rassert(workspace.number() >= workspace_size,
        7104202610, workspace.number(), workspace_size);
    rassert(workSize.cuBlockSize().x == 256
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

    ::prefix_sum_lookback_kernel
        <<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            input.cuptr(), output.cuptr(), states, next_tile, n);

    CUDA_CHECK_KERNEL_SYNC(stream);
}

}
