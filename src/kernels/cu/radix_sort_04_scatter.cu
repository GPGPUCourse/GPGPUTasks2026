#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void __launch_bounds__(GROUP_SIZE) radix_sort_04_scatter(
    const unsigned int* buffer1,
    const unsigned int* buffer2,
          unsigned int* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    static_assert(GROUP_SIZE == RADIX, "thread index is used as a digit");
    constexpr unsigned int WARPS = GROUP_SIZE / WARP_SIZE;
    __shared__ unsigned int tile_keys[TILE_SIZE];
    __shared__ unsigned int warp_counts[WARPS * RADIX];    __shared__ unsigned int scan[RADIX];
    __shared__ unsigned int global_base[RADIX];

    const unsigned int g = blockIdx.x;
    const unsigned int w = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int global_start = buffer2[g * RADIX + threadIdx.x];

    unsigned int key[KEYS_PER_THREAD];
    #pragma unroll
    for (unsigned int k = 0; k < KEYS_PER_THREAD; ++k) {
        const unsigned int i = g * TILE_SIZE + w * WARP_SIZE * KEYS_PER_THREAD + k * WARP_SIZE + lane;
        key[k] = i < a1 ? buffer1[i] : 0xFFFFFFFFu;
    }

    for (unsigned int d = lane; d < RADIX; d += WARP_SIZE)
        warp_counts[w * RADIX + d] = 0;
    __syncwarp();

    unsigned int rank[KEYS_PER_THREAD];
    #pragma unroll
    for (unsigned int k = 0; k < KEYS_PER_THREAD; ++k) {
        const unsigned int digit = (key[k] >> a2) & (RADIX - 1);
        unsigned int peers = 0xFFFFFFFFu;
        for (unsigned int b = 0; b < RADIX_BITS; ++b) {
            const unsigned int bit = (digit >> b) & 1;
            const unsigned int v = __ballot_sync(0xFFFFFFFFu, bit);
            peers &= bit ? v : ~v;
        }
        const unsigned int prev = warp_counts[w * RADIX + digit];
        __syncwarp();
        const unsigned int below = __popc(peers & ((1u << lane) - 1));
        if (below == 0)
            warp_counts[w * RADIX + digit] = prev + __popc(peers);
        __syncwarp();
        rank[k] = prev + below;
    }
    __syncthreads();

    const unsigned int d = threadIdx.x;
    unsigned int total = 0;
    for (unsigned int i = 0; i < WARPS; ++i) {
        const unsigned int c = warp_counts[i * RADIX + d];
        warp_counts[i * RADIX + d] = total;
        total += c;
    }

    scan[d] = total;
    __syncthreads();
    for (unsigned int offset = 1; offset < RADIX; offset *= 2) {
        const unsigned int v = d >= offset ? scan[d - offset] : 0;
        __syncthreads();
        scan[d] += v;
        __syncthreads();
    }
    const unsigned int tile_start = scan[d] - total;
    for (unsigned int i = 0; i < WARPS; ++i)
        warp_counts[i * RADIX + d] += tile_start;
    global_base[d] = global_start - tile_start;
    __syncthreads();

    #pragma unroll
    for (unsigned int k = 0; k < KEYS_PER_THREAD; ++k) {
        const unsigned int digit = (key[k] >> a2) & (RADIX - 1);
        tile_keys[warp_counts[w * RADIX + digit] + rank[k]] = key[k];
    }
    __syncthreads();

    const unsigned int tile_n = min(a1 - g * TILE_SIZE, (unsigned int) TILE_SIZE);
    for (unsigned int i = threadIdx.x; i < tile_n; i += GROUP_SIZE) {
        const unsigned int value = tile_keys[i];
        buffer3[global_base[(value >> a2) & (RADIX - 1)] + i] = value;
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
