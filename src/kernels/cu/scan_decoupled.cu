#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>

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

__device__ __forceinline__ unsigned int warp_scan_inclusive(unsigned int val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        unsigned int n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) val += n;
    }
    return val;
}

__launch_bounds__(BLOCK_THREADS, 8)
__global__ void scan_decoupled_kernel(const unsigned int* __restrict__ d_in,
                                      unsigned int* __restrict__ d_out,
                                      unsigned int* __restrict__ d_tile_counter,
                                      uint64_t* __restrict__ d_descriptors,
                                      unsigned int n,
                                      unsigned int num_tiles)
{
    __shared__ unsigned int s_tile_idx;
    __shared__ unsigned int s_tile_prefix;
    __shared__ unsigned int s_warp_sums[32];

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

    if (__builtin_expect(is_full_tile, 1)) {
        const uint4* in_u4 = reinterpret_cast<const uint4*>(d_in + tile_offset);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            u[c] = in_u4[c * 256 + tid];
            u[c].y += u[c].x;
            u[c].z += u[c].y;
            u[c].w += u[c].z;
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned int idx = tile_offset + c * 256 * 4 + tid * 4;
            u[c].x = (idx < n)     ? d_in[idx]     : 0;
            u[c].y = (idx + 1 < n) ? d_in[idx + 1] : 0;
            u[c].z = (idx + 2 < n) ? d_in[idx + 2] : 0;
            u[c].w = (idx + 3 < n) ? d_in[idx + 3] : 0;

            u[c].y += u[c].x;
            u[c].z += u[c].y;
            u[c].w += u[c].z;
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        unsigned int my_sum = u[c].w;
        unsigned int inc = warp_scan_inclusive(my_sum);
        unsigned int warp_exc = inc - my_sum;
        if (lane == 31) {
            s_warp_sums[c * 8 + warp] = inc;
        }
        u[c].x += warp_exc;
        u[c].y += warp_exc;
        u[c].z += warp_exc;
        u[c].w += warp_exc;
    }
    __syncthreads();

    unsigned int tile_agg = 0;
    if (warp == 0) {
        unsigned int val = s_warp_sums[lane];
        unsigned int scanned = warp_scan_inclusive(val);
        s_warp_sums[lane] = scanned;
        tile_agg = __shfl_sync(0xffffffff, scanned, 31);
    }

    if (tid == 0) {
        if (tile_idx == 0) {
            s_tile_prefix = 0;
            store_descriptor_release(&d_descriptors[0], FLAG_PREFIX, tile_agg);
        } else {
            store_descriptor_release(&d_descriptors[tile_idx], FLAG_AGGREGATE, tile_agg);

            unsigned int lookback_idx = tile_idx - 1;
            unsigned int exclusive_prefix = 0;

            while (true) {
                uint64_t probe = load_descriptor_relaxed(&d_descriptors[lookback_idx]);
                unsigned int status = (unsigned int)(probe >> 32);

                while (status == FLAG_NOT_READY) {
                    probe = load_descriptor_relaxed(&d_descriptors[lookback_idx]);
                    status = (unsigned int)(probe >> 32);
                }

                unsigned int val = (unsigned int)(probe & 0xFFFFFFFFu);
                exclusive_prefix += val;

                if (status == FLAG_PREFIX) {
                    break;
                }
                lookback_idx--;
            }

            s_tile_prefix = exclusive_prefix;
            store_descriptor_release(&d_descriptors[tile_idx], FLAG_PREFIX, exclusive_prefix + tile_agg);
        }
    }
    __syncthreads();

    const unsigned int tile_base = s_tile_prefix;

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const unsigned int warp_chunk_id = c * 8 + warp;
        const unsigned int warp_prefix = (warp_chunk_id > 0) ? s_warp_sums[warp_chunk_id - 1] : 0;
        const unsigned int thread_base = tile_base + warp_prefix;

        u[c].x += thread_base;
        u[c].y += thread_base;
        u[c].z += thread_base;
        u[c].w += thread_base;
    }

    if (__builtin_expect(is_full_tile, 1)) {
        uint4* out_u4 = reinterpret_cast<uint4*>(d_out + tile_offset);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            out_u4[c * 256 + tid] = u[c];
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            unsigned int idx = tile_offset + c * 256 * 4 + tid * 4;
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

    CUcontext ctx0 = nullptr;
    cuCtxGetCurrent(&ctx0);

    int dev_count = 1;
    cuDeviceGetCount(&dev_count);
    int num_gpus = std::min(dev_count, 4);

    const unsigned int num_tiles = (n + TILE_ITEMS - 1) / TILE_ITEMS;
    unsigned int* d_tile_counter = d_temp.cuptr();
    uint64_t* d_descriptors = reinterpret_cast<uint64_t*>(d_temp.cuptr() + DESCRIPTORS_OFFSET_U32);

    const size_t total_clear_bytes = (DESCRIPTORS_OFFSET_U32 * sizeof(unsigned int)) +
                                     (num_tiles * sizeof(uint64_t));

    cudaMemsetAsync(d_temp.cuptr(), 0, total_clear_bytes, context.cudaStream());

    if (num_gpus <= 1) {
        scan_decoupled_kernel<<<num_tiles, BLOCK_THREADS, 0, context.cudaStream()>>>(
            d_in.cuptr(),
            d_out.cuptr(),
            d_tile_counter,
            d_descriptors,
            n,
            num_tiles
        );
        CUDA_CHECK_KERNEL(context.cudaStream());
        return;
    }

    static bool p2p_initialized = false;
    static CUcontext contexts[4] = {nullptr};
    static CUstream streams[4] = {nullptr};

    if (!p2p_initialized) {
        contexts[0] = ctx0;
        streams[0] = (CUstream)context.cudaStream();

        for (int i = 1; i < num_gpus; ++i) {
            CUdevice dev_i;
            cuDeviceGet(&dev_i, i);

            CUresult res;
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)
            CUctxCreateParams params{};
            res = cuCtxCreate(&contexts[i], &params, 0, dev_i);
#else
            res = cuCtxCreate(&contexts[i], 0, dev_i);
#endif
            if (res != CUDA_SUCCESS) {
                num_gpus = i;
                break;
            }
            cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING);

            cuCtxSetCurrent(contexts[i]);
            res = cuCtxEnablePeerAccess(ctx0, 0);
            if (res != CUDA_SUCCESS && res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                std::cerr << "Warning: cuCtxEnablePeerAccess from GPU " << i << " to GPU 0 failed: " << res << std::endl;
            }

            cuCtxSetCurrent(ctx0);
            res = cuCtxEnablePeerAccess(contexts[i], 0);
            if (res != CUDA_SUCCESS && res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
                std::cerr << "Warning: cuCtxEnablePeerAccess from GPU 0 to GPU " << i << " failed: " << res << std::endl;
            }
        }
        p2p_initialized = true;
    }

    cuCtxSetCurrent(ctx0);
    cuStreamSynchronize(streams[0]);

    unsigned int tiles_per_gpu = num_tiles / num_gpus;
    unsigned int remainder = num_tiles % num_gpus;

    for (int i = 0; i < num_gpus; ++i) {
        unsigned int grid_size = tiles_per_gpu + (i < remainder ? 1 : 0);
        if (grid_size == 0) continue;

        cuCtxSetCurrent(contexts[i]);

        scan_decoupled_kernel<<<grid_size, BLOCK_THREADS, 0, (cudaStream_t)streams[i]>>>(
            d_in.cuptr(),
            d_out.cuptr(),
            d_tile_counter,
            d_descriptors,
            n,
            num_tiles
        );
    }

    for (int i = 0; i < num_gpus; ++i) {
        cuCtxSetCurrent(contexts[i]);
        cuStreamSynchronize(streams[i]);
    }

    cuCtxSetCurrent(ctx0);
}
} // namespace cuda