#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include <cuda_runtime.h>
#include <algorithm>

#include "helpers/rassert.cu"
#include "../defines.h"

#define RADIX_BITS        8
#define RADIX_BINS        256
#define BLOCK_THREADS     256
#define ITEMS_PER_THREAD  16
#define TILE_ITEMS        (BLOCK_THREADS * ITEMS_PER_THREAD) // 4096

#define FLAG_NOT_READY    0u
#define FLAG_AGGREGATE    1u
#define FLAG_PREFIX       2u

#define HEADER_OFFSET_U32 1032

#define WARP_HIST_PADDED_STRIDE 264
#define S_VALS_PADDED_SIZE      4224

__device__ __forceinline__ unsigned int pad_s_vals(unsigned int idx) {
    return idx + (idx >> 5);
}

__device__ __forceinline__ unsigned int pad_bin(unsigned int d) {
    return d + (d >> 5);
}

__device__ __forceinline__ void store_descriptor_cg(unsigned int* ptr, unsigned int status, unsigned int val) {
    unsigned int d = (status << 30) | val;
    asm volatile("st.global.cg.b32 [%0], %1;" :: "l"(ptr), "r"(d) : "memory");
}

__device__ __forceinline__ unsigned int load_descriptor_cg(const unsigned int* ptr) {
    unsigned int d;
    asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(d) : "l"(ptr) : "memory");
    return d;
}

__device__ __forceinline__ unsigned int ptx_match_any(unsigned int val) {
    unsigned int r;
    asm volatile("match.any.sync.b32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(val));
    return r;
}


__launch_bounds__(BLOCK_THREADS, 5)
__global__ void global_histogram_kernel(
    const unsigned int* __restrict__ d_in,
          unsigned int* __restrict__ d_aux,
    unsigned int n)
{
    __shared__ unsigned int s_hist[4][RADIX_BINS + 1];
    __shared__ unsigned int s_warp_sums[8];
    __shared__ bool s_is_last_block;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;

    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        s_hist[p][tid] = 0u;
    }
    __syncthreads();

    const unsigned int u4_n = n / 4;
    const uint4* in_u4 = reinterpret_cast<const uint4*>(d_in);
    const unsigned int stride = gridDim.x * BLOCK_THREADS;

    for (unsigned int idx = blockIdx.x * BLOCK_THREADS + tid; idx < u4_n; idx += stride) {
        uint4 v = in_u4[idx];

        atomicAdd(&s_hist[0][ v.x        & 0xFFu], 1u);
        atomicAdd(&s_hist[1][(v.x >>  8) & 0xFFu], 1u);
        atomicAdd(&s_hist[2][(v.x >> 16) & 0xFFu], 1u);
        atomicAdd(&s_hist[3][ v.x >> 24         ], 1u);

        atomicAdd(&s_hist[0][ v.y        & 0xFFu], 1u);
        atomicAdd(&s_hist[1][(v.y >>  8) & 0xFFu], 1u);
        atomicAdd(&s_hist[2][(v.y >> 16) & 0xFFu], 1u);
        atomicAdd(&s_hist[3][ v.y >> 24         ], 1u);

        atomicAdd(&s_hist[0][ v.z        & 0xFFu], 1u);
        atomicAdd(&s_hist[1][(v.z >>  8) & 0xFFu], 1u);
        atomicAdd(&s_hist[2][(v.z >> 16) & 0xFFu], 1u);
        atomicAdd(&s_hist[3][ v.z >> 24         ], 1u);

        atomicAdd(&s_hist[0][ v.w        & 0xFFu], 1u);
        atomicAdd(&s_hist[1][(v.w >>  8) & 0xFFu], 1u);
        atomicAdd(&s_hist[2][(v.w >> 16) & 0xFFu], 1u);
        atomicAdd(&s_hist[3][ v.w >> 24         ], 1u);
    }

    if (blockIdx.x == 0) {
        for (unsigned int i = u4_n * 4 + tid; i < n; i += BLOCK_THREADS) {
            unsigned int val = d_in[i];
            atomicAdd(&s_hist[0][ val        & 0xFFu], 1u);
            atomicAdd(&s_hist[1][(val >>  8) & 0xFFu], 1u);
            atomicAdd(&s_hist[2][(val >> 16) & 0xFFu], 1u);
            atomicAdd(&s_hist[3][ val >> 24         ], 1u);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        unsigned int sum = s_hist[p][tid];
        if (sum > 0u) {
            atomicAdd(&d_aux[p * RADIX_BINS + tid], sum);
        }
    }
    __threadfence();
    __syncthreads();

    if (tid == 0) {
        unsigned int done = atomicAdd(&d_aux[1028], 1u);
        s_is_last_block = (done == gridDim.x - 1u);
    }
    __syncthreads();

    if (s_is_last_block) {
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            unsigned int val = d_aux[p * RADIX_BINS + tid];
            unsigned int inc = val;

            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                unsigned int tmp = __shfl_up_sync(0xffffffff, inc, offset);
                if (lane >= offset) inc += tmp;
            }
            if (lane == 31) s_warp_sums[warp] = inc;
            __syncthreads();

            if (warp == 0) {
                unsigned int ws = (lane < 8) ? s_warp_sums[lane] : 0u;
                #pragma unroll
                for (int offset = 1; offset < 8; offset <<= 1) {
                    unsigned int tmp = __shfl_up_sync(0xffffffff, ws, offset);
                    if (lane >= offset) ws += tmp;
                }
                if (lane < 8) s_warp_sums[lane] = ws;
            }
            __syncthreads();

            unsigned int warp_prefix = (warp > 0) ? s_warp_sums[warp - 1] : 0u;
            d_aux[p * RADIX_BINS + tid] = warp_prefix + (inc - val);
            __syncthreads();
        }
    }
}


__launch_bounds__(BLOCK_THREADS, 4)
__global__ void onesweep_pass_kernel(
    const unsigned int* __restrict__ d_in,
          unsigned int* __restrict__ d_aux,
          unsigned int* __restrict__ d_out,
    unsigned int shift,
    unsigned int n)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31;
    const unsigned int num_warps = BLOCK_THREADS / 32;
    const unsigned int pass = shift >> 3;
    const unsigned int num_tiles = gridDim.x;
    const unsigned int lane_mask_lt = (1u << lane) - 1u;

    __shared__ unsigned int s_tile_idx;
    __shared__ unsigned int s_warp_sums[8];
    __shared__ unsigned int s_global_offset[RADIX_BINS];
    __shared__ union {
        unsigned int warp_hist[BLOCK_THREADS / 32][WARP_HIST_PADDED_STRIDE];
        unsigned int s_vals[S_VALS_PADDED_SIZE];
    } shared_mem;

    if (tid == 0) {
        s_tile_idx = atomicAdd(&d_aux[1024 + pass], 1u);
    }
    #pragma unroll
    for (unsigned int w = 0; w < num_warps; ++w) {
        shared_mem.warp_hist[w][pad_bin(tid)] = 0u;
    }
    __syncthreads();

    const unsigned int tile_idx = s_tile_idx;
    const unsigned int tile_offset = tile_idx * TILE_ITEMS;
    const unsigned int warp_offset = tile_offset + warp * (32 * ITEMS_PER_THREAD);
    const bool is_full_tile = (tile_offset + TILE_ITEMS <= n);

    unsigned int* next_descriptors = &d_aux[HEADER_OFFSET_U32 + ((pass + 1u) & 1u) * num_tiles * RADIX_BINS];
    store_descriptor_cg(&next_descriptors[tile_idx * RADIX_BINS + tid], 0u, 0u);

    unsigned int vals[ITEMS_PER_THREAD];
    unsigned int ranks[ITEMS_PER_THREAD];
    volatile unsigned int* my_warp_hist = shared_mem.warp_hist[warp];

    if (__builtin_expect(is_full_tile, 1)) {
        #pragma unroll
        for (int s = 0; s < ITEMS_PER_THREAD; ++s) {
            vals[s] = d_in[warp_offset + s * 32 + lane];
        }

        #pragma unroll
        for (int s = 0; s < ITEMS_PER_THREAD; s += 2) {
            const unsigned int d0 = (vals[s] >> shift) & 0xFFu;
            const unsigned int d1 = (vals[s + 1] >> shift) & 0xFFu;

            const unsigned int m0 = ptx_match_any(d0);
            const unsigned int m1 = ptx_match_any(d1);

            const unsigned int p0 = my_warp_hist[pad_bin(d0)];
            const unsigned int rank_in_warp_0 = __popc(m0 & lane_mask_lt);
            ranks[s] = p0 + rank_in_warp_0;
            if (rank_in_warp_0 == 0) {
                my_warp_hist[pad_bin(d0)] = p0 + __popc(m0);
            }
            __syncwarp();

            const unsigned int p1 = my_warp_hist[pad_bin(d1)];
            const unsigned int rank_in_warp_1 = __popc(m1 & lane_mask_lt);
            ranks[s + 1] = p1 + rank_in_warp_1;
            if (rank_in_warp_1 == 0) {
                my_warp_hist[pad_bin(d1)] = p1 + __popc(m1);
            }
            __syncwarp();
        }
    } else {
        #pragma unroll
        for (int s = 0; s < ITEMS_PER_THREAD; ++s) {
            const unsigned int idx = warp_offset + s * 32 + lane;
            const bool is_valid = (idx < n);
            const unsigned int v = is_valid ? d_in[idx] : 0u;
            const unsigned int d = (v >> shift) & 0xFFu;
            vals[s] = v;

            const unsigned int match_key = is_valid ? d : (0x100u + lane);
            const unsigned int match_mask = ptx_match_any(match_key);
            const unsigned int prev_count = is_valid ? my_warp_hist[pad_bin(d)] : 0u;
            const unsigned int rank_in_warp = __popc(match_mask & lane_mask_lt);
            ranks[s] = prev_count + rank_in_warp;
            if (is_valid && (rank_in_warp == 0)) {
                my_warp_hist[pad_bin(d)] = prev_count + __popc(match_mask);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    unsigned int tile_agg = 0u;
    #pragma unroll
    for (unsigned int w = 0; w < num_warps; ++w) {
        const unsigned int cnt = shared_mem.warp_hist[w][pad_bin(tid)];
        shared_mem.warp_hist[w][pad_bin(tid)] = tile_agg;
        tile_agg += cnt;
    }

    unsigned int* d_descriptors = &d_aux[HEADER_OFFSET_U32 + (pass & 1u) * num_tiles * RADIX_BINS];
    unsigned int global_base;
    if (tile_idx == 0u) {
        store_descriptor_cg(&d_descriptors[tid], FLAG_PREFIX, tile_agg);
        global_base = d_aux[pass * RADIX_BINS + tid];
    } else {
        store_descriptor_cg(&d_descriptors[tile_idx * RADIX_BINS + tid], FLAG_AGGREGATE, tile_agg);

        int lookback_idx = static_cast<int>(tile_idx) - 1;
        unsigned int exclusive_prefix = 0u;

        while (lookback_idx >= 0) {
            unsigned int probe = load_descriptor_cg(&d_descriptors[lookback_idx * RADIX_BINS + tid]);
            unsigned int status = probe >> 30;

            while (status == FLAG_NOT_READY) {
                __nanosleep(32);
                probe = load_descriptor_cg(&d_descriptors[lookback_idx * RADIX_BINS + tid]);
                status = probe >> 30;
            }

            exclusive_prefix += (probe & 0x3FFFFFFFu);
            if (status == FLAG_PREFIX) {
                break;
            }
            lookback_idx--;
        }

        store_descriptor_cg(&d_descriptors[tile_idx * RADIX_BINS + tid], FLAG_PREFIX, exclusive_prefix + tile_agg);
        global_base = d_aux[pass * RADIX_BINS + tid] + exclusive_prefix;
    }

    __syncthreads();

    unsigned int inc = tile_agg;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int tmp = __shfl_up_sync(0xffffffff, inc, offset);
        if (lane >= offset) inc += tmp;
    }
    if (lane == 31) s_warp_sums[warp] = inc;
    __syncthreads();

    unsigned int ws = (lane < 8) ? s_warp_sums[lane] : 0u;
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        unsigned int tmp = __shfl_up_sync(0xffffffff, ws, offset);
        if (lane >= offset) ws += tmp;
    }
    const unsigned int warp_prefix = (warp > 0) ? __shfl_sync(0xffffffff, ws, warp - 1) : 0u;
    const unsigned int my_bucket_start = warp_prefix + (inc - tile_agg);

    s_global_offset[tid] = global_base - my_bucket_start;
    #pragma unroll
    for (unsigned int w = 0; w < num_warps; ++w) {
        shared_mem.warp_hist[w][pad_bin(tid)] += my_bucket_start;
    }
    __syncthreads();

    #pragma unroll
    for (int s = 0; s < ITEMS_PER_THREAD; ++s) {
        const unsigned int d = (vals[s] >> shift) & 0xFFu;
        ranks[s] += shared_mem.warp_hist[warp][pad_bin(d)];
    }
    __syncthreads();

    if (__builtin_expect(is_full_tile, 1)) {
        #pragma unroll
        for (int s = 0; s < ITEMS_PER_THREAD; ++s) {
            shared_mem.s_vals[pad_s_vals(ranks[s])] = vals[s];
        }
        __syncthreads();

        #pragma unroll
        for (int s = 0; s < ITEMS_PER_THREAD; ++s) {
            const unsigned int pos = s * BLOCK_THREADS + tid;
            const unsigned int v = shared_mem.s_vals[pad_s_vals(pos)];
            const unsigned int d = (v >> shift) & 0xFFu;
            d_out[s_global_offset[d] + pos] = v;
        }
    } else {
        #pragma unroll
        for (int s = 0; s < ITEMS_PER_THREAD; ++s) {
            const unsigned int idx = warp_offset + s * 32 + lane;
            if (idx < n) {
                shared_mem.s_vals[pad_s_vals(ranks[s])] = vals[s];
            }
        }
        __syncthreads();

        const unsigned int valid_in_tile = (tile_offset < n) ? (n - tile_offset) : 0u;
        for (unsigned int pos = tid; pos < valid_in_tile; pos += BLOCK_THREADS) {
            const unsigned int v = shared_mem.s_vals[pad_s_vals(pos)];
            const unsigned int d = (v >> shift) & 0xFFu;
            d_out[s_global_offset[d] + pos] = v;
        }
    }
}


namespace cuda {

void radix_sort_04_scatter(const gpu::WorkSize & /*workSize*/,
    const gpu::gpu_mem_32u &d_in,
    const gpu::gpu_mem_32u &d_buf1_const,
    gpu::gpu_mem_32u &d_out,
    unsigned int n,
    unsigned int /*unused*/)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    static int num_sms = 0;
    if (num_sms == 0) {
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        cudaFuncSetAttribute(onesweep_pass_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
    }

    unsigned int* d_ping_pong = const_cast<unsigned int*>(d_buf1_const.cuptr());
    const unsigned int num_tiles = (n + TILE_ITEMS - 1) / TILE_ITEMS;
    const size_t aux_words = HEADER_OFFSET_U32 + 2 * num_tiles * RADIX_BINS;

    static unsigned int* d_aux = nullptr;
    static size_t allocated_words = 0;
    if (aux_words > allocated_words) {
        if (d_aux) cudaFree(d_aux);
        cudaMalloc(&d_aux, aux_words * sizeof(unsigned int));
        cudaMemset(d_aux, 0, aux_words * sizeof(unsigned int));
        allocated_words = aux_words;
    }

    cudaMemsetAsync(d_aux, 0, HEADER_OFFSET_U32 * sizeof(unsigned int), stream);

    const unsigned int hist_blocks = std::min(num_tiles, static_cast<unsigned int>(num_sms * 16));
    global_histogram_kernel<<<hist_blocks, BLOCK_THREADS, 0, stream>>>(
        d_in.cuptr(), d_aux, n
    );

    onesweep_pass_kernel<<<num_tiles, BLOCK_THREADS, 0, stream>>>(
        d_in.cuptr(), d_aux, d_ping_pong, 0, n
    );
    onesweep_pass_kernel<<<num_tiles, BLOCK_THREADS, 0, stream>>>(
        d_ping_pong, d_aux, d_out.cuptr(), 8, n
    );
    onesweep_pass_kernel<<<num_tiles, BLOCK_THREADS, 0, stream>>>(
        d_out.cuptr(), d_aux, d_ping_pong, 16, n
    );
    onesweep_pass_kernel<<<num_tiles, BLOCK_THREADS, 0, stream>>>(
        d_ping_pong, d_aux, d_out.cuptr(), 24, n
    );

    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda