#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define FLAG_AGGREGATE 1u
#define FLAG_INCLUSIVE 2u
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ uint4 load_vec(const unsigned int* input, unsigned int i, unsigned int n)
{
    if (i + 4 <= n)
        return __ldcs(reinterpret_cast<const uint4*>(input + i));
    uint4 v = make_uint4(0, 0, 0, 0);
    if (i + 0 < n) v.x = input[i + 0];
    if (i + 1 < n) v.y = input[i + 1];
    if (i + 2 < n) v.z = input[i + 2];
    if (i + 3 < n) v.w = input[i + 3];
    return v;
}

__device__ __forceinline__ void store_vec(unsigned int* output, unsigned int i, unsigned int n, uint4 v)
{
    if (i + 4 <= n) {
        __stcs(reinterpret_cast<uint4*>(output + i), v);
        return;
    }
    if (i + 0 < n) output[i + 0] = v.x;
    if (i + 1 < n) output[i + 1] = v.y;
    if (i + 2 < n) output[i + 2] = v.z;
    if (i + 3 < n) output[i + 3] = v.w;
}

__device__ __forceinline__ unsigned int warp_sum(unsigned int x)
{
    for (int offset = 16; offset > 0; offset /= 2)
        x += __shfl_xor_sync(FULL_MASK, x, offset);
    return x;
}

__device__ __forceinline__ void store_status(unsigned long long* status, unsigned int tile, unsigned int flag, unsigned int value)
{
    *reinterpret_cast<volatile unsigned long long*>(status + tile) = (static_cast<unsigned long long>(flag) << 32) | value;
}

__global__ void __launch_bounds__(GROUP_SIZE) prefix_sum_02_prefix_accumulation(
    const unsigned int* input,
          unsigned int* prefix_sum_accum,
    unsigned long long* status,
          unsigned int* tile_counter,
    unsigned int n)
{
    constexpr unsigned int NWARPS = GROUP_SIZE / 32;

    __shared__ unsigned int s_tile;
    __shared__ unsigned int s_exclusive;
    __shared__ unsigned int s_warp_sums[PREFIX_SUM_VEC_PER_THREAD][NWARPS];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp = tid / 32;

    if (tid == 0)
        s_tile = atomicAdd(tile_counter, 1);
    __syncthreads();
    const unsigned int tile = s_tile;
    const unsigned int tile_base = tile * PREFIX_SUM_TILE_SIZE;

    uint4 v[PREFIX_SUM_VEC_PER_THREAD];
    unsigned int scan[PREFIX_SUM_VEC_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < PREFIX_SUM_VEC_PER_THREAD; ++r) {
        v[r] = load_vec(input, tile_base + (r * GROUP_SIZE + tid) * 4, n);
        v[r].y += v[r].x;
        v[r].z += v[r].y;
        v[r].w += v[r].z;
        scan[r] = v[r].w;
    }

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        #pragma unroll
        for (int r = 0; r < PREFIX_SUM_VEC_PER_THREAD; ++r) {
            unsigned int x = __shfl_up_sync(FULL_MASK, scan[r], offset);
            if (lane >= offset)
                scan[r] += x;
        }
    }

    if (lane == 31) {
        #pragma unroll
        for (int r = 0; r < PREFIX_SUM_VEC_PER_THREAD; ++r)
            s_warp_sums[r][warp] = scan[r];
    }
    __syncthreads();

    unsigned int thread_offset[PREFIX_SUM_VEC_PER_THREAD];
    unsigned int aggregate = 0;
    #pragma unroll
    for (int r = 0; r < PREFIX_SUM_VEC_PER_THREAD; ++r) {
        unsigned int warp_prefix = 0;
        unsigned int chunk_total = 0;
        #pragma unroll
        for (unsigned int w = 0; w < NWARPS; ++w) {
            unsigned int x = s_warp_sums[r][w];
            if (w < warp)
                warp_prefix += x;
            chunk_total += x;
        }
        thread_offset[r] = aggregate + warp_prefix + scan[r] - v[r].w;
        aggregate += chunk_total;
    }

    if (warp == 0) {
        unsigned int exclusive = 0;
        if (tile == 0) {
            if (lane == 0)
                store_status(status, 0, FLAG_INCLUSIVE, aggregate);
        } else {
            if (lane == 0)
                store_status(status, tile, FLAG_AGGREGATE, aggregate);

            int window = (int) tile - 1;
            while (true) {
                const int pred = window - (int) lane;
                unsigned int flag = FLAG_INCLUSIVE;
                unsigned int value = 0;
                if (pred >= 0) {
                    const volatile unsigned long long* p = status + pred;
                    unsigned long long st;
                    do {
                        st = *p;
                    } while ((st >> 32) == 0);
                    flag = (unsigned int) (st >> 32);
                    value = (unsigned int) st;
                }

                const unsigned int inclusive_mask = __ballot_sync(FULL_MASK, flag == FLAG_INCLUSIVE);
                if (inclusive_mask) {
                    const unsigned int first = __ffs(inclusive_mask) - 1;
                    if (lane > first)
                        value = 0;
                    exclusive += warp_sum(value);
                    break;
                }
                exclusive += warp_sum(value);
                window -= 32;
            }

            if (lane == 0)
                store_status(status, tile, FLAG_INCLUSIVE, exclusive + aggregate);
        }
        if (lane == 0)
            s_exclusive = exclusive;
    }
    __syncthreads();
    const unsigned int exclusive = s_exclusive;

    #pragma unroll
    for (int r = 0; r < PREFIX_SUM_VEC_PER_THREAD; ++r) {
        const unsigned int add = exclusive + thread_offset[r];
        v[r].x += add;
        v[r].y += add;
        v[r].z += add;
        v[r].w += add;
        store_vec(prefix_sum_accum, tile_base + (r * GROUP_SIZE + tid) * 4, n, v[r]);
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &prefix_sum_accum, gpu::gpu_mem_32u &tiles_status, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    const unsigned int ntiles = div_ceil(n, (unsigned int) PREFIX_SUM_TILE_SIZE);
    rassert(tiles_status.size() >= 2 * ntiles + 1, 7456234523412, tiles_status.size(), ntiles);
    unsigned long long* status = reinterpret_cast<unsigned long long*>(tiles_status.cuptr());
    unsigned int* tile_counter = tiles_status.cuptr() + 2 * ntiles;
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), prefix_sum_accum.cuptr(), status, tile_counter, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
