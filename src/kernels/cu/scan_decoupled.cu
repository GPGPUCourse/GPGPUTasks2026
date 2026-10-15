#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define TILE_ITEMS 4096
#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 16

#define FLAG_NOT_READY 0u
#define FLAG_AGGREGATE 1u
#define FLAG_PREFIX    2u

#define DESCRIPTORS_OFFSET_U32 32

__device__ __forceinline__ void store_descriptor_release(uint64_t* ptr, unsigned int status, unsigned int val) {
    uint64_t d = (((uint64_t)status) << 32) | (uint64_t)val;
    asm volatile("st.release.gpu.u64 [%0], %1;" :: "l"(ptr), "l"(d) : "memory");
}

__device__ __forceinline__ uint64_t load_descriptor_relaxed(const uint64_t* ptr) {
    uint64_t d;
    asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(d) : "l"(ptr) : "memory");
    return d;
}

__device__ __forceinline__ void load_descriptor_acquire(const uint64_t* ptr, unsigned int &status, unsigned int &val) {
    uint64_t d;
    asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(d) : "l"(ptr) : "memory");
    status = (unsigned int)(d >> 32);
    val    = (unsigned int)(d & 0xFFFFFFFFu);
}

__device__ __forceinline__ unsigned int warp_scan_inclusive(unsigned int val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}

__launch_bounds__(BLOCK_THREADS, 2)
__global__ void scan_decoupled_kernel(const unsigned int* __restrict__ d_in,
                                      unsigned int* __restrict__ d_out,
                                      unsigned int* __restrict__ d_tile_counter,
                                      uint64_t* __restrict__ d_descriptors,
                                      unsigned int n,
                                      unsigned int num_tiles)
{
    __shared__ unsigned int s_tile_idx;
    __align__(16) __shared__ unsigned int s_base[4];
    __shared__ unsigned int s_warp_chunk_sums[4][8];

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;

    if (tid == 0) {
        s_tile_idx = atomicAdd(d_tile_counter, 1);
    }
    __syncthreads();

    const unsigned int tile_idx = s_tile_idx;
    if (tile_idx >= num_tiles) return;

    const unsigned int tile_offset = tile_idx * TILE_ITEMS;
    const bool is_full_tile = (tile_offset + TILE_ITEMS <= n);

    uint4 u[4];
    unsigned int chunk_sum[4];
    unsigned int chunk_exc[4];

    if (is_full_tile) {
        const uint4* in_u4 = reinterpret_cast<const uint4*>(d_in + tile_offset);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            u[c] = in_u4[c * 256 + tid];
            u[c].y += u[c].x;
            u[c].z += u[c].y;
            u[c].w += u[c].z;
            chunk_sum[c] = u[c].w;
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned int idx = tile_offset + c * 256 * 4 + tid * 4;
            u[c].x = (idx < n) ? d_in[idx] : 0;
            u[c].y = (idx + 1 < n) ? d_in[idx + 1] : 0;
            u[c].z = (idx + 2 < n) ? d_in[idx + 2] : 0;
            u[c].w = (idx + 3 < n) ? d_in[idx + 3] : 0;

            u[c].y += u[c].x;
            u[c].z += u[c].y;
            u[c].w += u[c].z;
            chunk_sum[c] = u[c].w;
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        unsigned int inc = warp_scan_inclusive(chunk_sum[c]);
        chunk_exc[c] = inc - chunk_sum[c];
        if (lane == 31) {
            s_warp_chunk_sums[c][warp] = inc;
        }
    }
    __syncthreads();

    if (tid == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned int sum = s_warp_chunk_sums[c][0];
        //    #pragma unroll
            for (int w = 1; w < 8; ++w) {
                sum += s_warp_chunk_sums[c][w];
                s_warp_chunk_sums[c][w] = sum;
            }
        }

        unsigned int tile_agg = s_warp_chunk_sums[0][7] + s_warp_chunk_sums[1][7] +
                               s_warp_chunk_sums[2][7] + s_warp_chunk_sums[3][7];

        if (tile_idx == 0) {
            s_base[0] = 0;
            s_base[1] = s_warp_chunk_sums[0][7];
            s_base[2] = s_base[1] + s_warp_chunk_sums[1][7];
            s_base[3] = s_base[2] + s_warp_chunk_sums[2][7];
            store_descriptor_release(&d_descriptors[0], FLAG_PREFIX, tile_agg);
        } else {
            store_descriptor_release(&d_descriptors[tile_idx], FLAG_AGGREGATE, tile_agg);

            unsigned int lookback_idx = tile_idx - 1;
            unsigned int exclusive_prefix = 0;

            while (true) {
                uint64_t probe = load_descriptor_relaxed(&d_descriptors[lookback_idx]);
                unsigned int status = (unsigned int)(probe >> 32);

                while (status == FLAG_NOT_READY) {
                    __nanosleep(32);
                    probe = load_descriptor_relaxed(&d_descriptors[lookback_idx]);
                    status = (unsigned int)(probe >> 32);
                }

                unsigned int val;
                load_descriptor_acquire(&d_descriptors[lookback_idx], status, val);

                if (status == FLAG_PREFIX) {
                    exclusive_prefix += val;
                    break;
                } else {
                    exclusive_prefix += val;
                    lookback_idx--;
                }
            }

            s_base[0] = exclusive_prefix;
            s_base[1] = exclusive_prefix + s_warp_chunk_sums[0][7];
            s_base[2] = s_base[1] + s_warp_chunk_sums[1][7];
            s_base[3] = s_base[2] + s_warp_chunk_sums[2][7];
            store_descriptor_release(&d_descriptors[tile_idx], FLAG_PREFIX, exclusive_prefix + tile_agg);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (warp > 0) {
            chunk_exc[c] += s_warp_chunk_sums[c][warp - 1];
        }
    }

    const uint4 b = *reinterpret_cast<const uint4*>(s_base);
    const unsigned int base[4] = {b.x, b.y, b.z, b.w};

    if (is_full_tile) {
        uint4* out_u4 = reinterpret_cast<uint4*>(d_out + tile_offset);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned int thread_base = base[c] + chunk_exc[c];
            u[c].x += thread_base;
            u[c].y += thread_base;
            u[c].z += thread_base;
            u[c].w += thread_base;
            out_u4[c * 256 + tid] = u[c];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned int idx = tile_offset + c * 256 * 4 + tid * 4;
            unsigned int thread_base = base[c] + chunk_exc[c];
            u[c].x += thread_base;
            u[c].y += thread_base;
            u[c].z += thread_base;
            u[c].w += thread_base;

            if (idx < n)     d_out[idx]     = u[c].x;
            if (idx + 1 < n) d_out[idx + 1] = u[c].y;
            if (idx + 2 < n) d_out[idx + 2] = u[c].z;
            if (idx + 3 < n) d_out[idx + 3] = u[c].w;
        }
    }
}

namespace cuda {
void scan_decoupled(const gpu::gpu_mem_32u &d_in,
                    gpu::gpu_mem_32u &d_out,
                    gpu::gpu_mem_32u &d_temp,
                    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    const unsigned int num_tiles = (n + TILE_ITEMS - 1) / TILE_ITEMS;

    unsigned int* d_tile_counter = d_temp.cuptr();
    uint64_t* d_descriptors = reinterpret_cast<uint64_t*>(d_temp.cuptr() + DESCRIPTORS_OFFSET_U32);

    const size_t total_clear_bytes = (DESCRIPTORS_OFFSET_U32 * sizeof(unsigned int)) +
                                     (num_tiles * sizeof(uint64_t));
    cudaMemsetAsync(d_temp.cuptr(), 0, total_clear_bytes, stream);

    scan_decoupled_kernel<<<num_tiles, BLOCK_THREADS, 0, stream>>>(
        d_in.cuptr(),
        d_out.cuptr(),
        d_tile_counter,
        d_descriptors,
        n,
        num_tiles
    );

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda