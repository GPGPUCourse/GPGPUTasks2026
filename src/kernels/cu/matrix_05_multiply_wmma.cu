#include "../defines.h"
#include "helpers/rassert.cu"
#include <cuda_fp16.h>
#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>
#include <mma.h>
#include <cstdio>

__global__ void matrix_multiply_prepare_half(
    const float* __restrict__ a,
    const float* __restrict__ b,
    half* __restrict__ a_packed,
    half* __restrict__ b_packed,
    unsigned int w,
    unsigned int h,
    unsigned int k,
    unsigned int packed_k,
    unsigned int a_blocks,
    unsigned int b_column_tiles)
{
    const unsigned int thread = threadIdx.x;
    if (blockIdx.x < a_blocks) {
        const size_t index = (size_t(blockIdx.x) * MATRIX_WMMA_PACK_THREADS + thread) * 4;
        const unsigned int row = index / packed_k;
        const unsigned int col = index % packed_k;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < h) {
            const float* input = a + size_t(row) * k;
            if ((k & 3) == 0 && col + 3 < k) {
                value = *reinterpret_cast<const float4*>(input + col);
            } else {
                value.x = col < k ? input[col] : 0.0f;
                value.y = col + 1 < k ? input[col + 1] : 0.0f;
                value.z = col + 2 < k ? input[col + 2] : 0.0f;
                value.w = col + 3 < k ? input[col + 3] : 0.0f;
            }
        }
        half2* output = reinterpret_cast<half2*>(a_packed + index);
        output[0] = __floats2half2_rn(value.x, value.y);
        output[1] = __floats2half2_rn(value.z, value.w);
    } else {
        __shared__ float tile[32][33];
        const unsigned int tx = thread % 32;
        const unsigned int ty = thread / 32;
        const unsigned int tile_index = blockIdx.x - a_blocks;
        const unsigned int block_col = (tile_index % b_column_tiles) * 32;
        const unsigned int block_k = (tile_index / b_column_tiles) * 32;
#pragma unroll
        for (unsigned int j = 0; j < 32; j += 8) {
            const unsigned int row = block_k + ty + j;
            const unsigned int col = block_col + tx;
            tile[ty + j][tx] = row < k && col < w ? b[size_t(row) * w + col] : 0.0f;
        }
        __syncthreads();
        const unsigned int pair = tx & 15;
        const unsigned int row = 2 * ty + (tx >> 4);
#pragma unroll
        for (unsigned int j = 0; j < 32; j += 16) {
            const half2 value = __floats2half2_rn(tile[2 * pair][row + j], tile[2 * pair + 1][row + j]);
            half2* output = reinterpret_cast<half2*>(b_packed + size_t(block_col + row + j) * packed_k + block_k);
            output[pair] = value;
        }
    }
}

__global__ void matrix_multiply_wmma(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    unsigned int w,
    unsigned int h,
    unsigned int packed_k)
{
    constexpr unsigned int TILE = 16;
    constexpr unsigned int BM = MATRIX_WMMA_BLOCK_M;
    constexpr unsigned int BN = MATRIX_WMMA_BLOCK_N;
    constexpr unsigned int BK = MATRIX_WMMA_BLOCK_K;
    constexpr unsigned int LD = BK + 16;
    constexpr unsigned int WARPS = MATRIX_WMMA_WARPS;
    constexpr unsigned int ROW_VECTORS = BK / 8;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;

    __shared__ __align__(32) unsigned char shared_bytes[(BM + BN) * LD * sizeof(half)];
    half* a_half = reinterpret_cast<half*>(shared_bytes);
    half* b_half = a_half + BM * LD;
    float (*c_float)[TILE * TILE] = reinterpret_cast<float (*)[TILE * TILE]>(shared_bytes);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2][4];
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
            nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    for (unsigned int k0 = 0; k0 < packed_k; k0 += BK) {
#pragma unroll
        for (unsigned int index = thread; index < BM * ROW_VECTORS; index += 32 * WARPS) {
            const unsigned int row = index / ROW_VECTORS;
            const unsigned int vector = index % ROW_VECTORS;
            const size_t source = size_t(block_row + row) * (packed_k / 8) + k0 / 8 + vector;
            reinterpret_cast<uint4*>(a_half)[row * (LD / 8) + vector] = reinterpret_cast<const uint4*>(a)[source];
        }
#pragma unroll
        for (unsigned int index = thread; index < BN * ROW_VECTORS; index += 32 * WARPS) {
            const unsigned int row = index / ROW_VECTORS;
            const unsigned int vector = index % ROW_VECTORS;
            const size_t source = size_t(block_col + row) * (packed_k / 8) + k0 / 8 + vector;
            reinterpret_cast<uint4*>(b_half)[row * (LD / 8) + vector] = reinterpret_cast<const uint4*>(b)[source];
        }
        __syncthreads();
#pragma unroll
        for (unsigned int kk = 0; kk < BK; kk += TILE) {
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i)
                nvcuda::wmma::load_matrix_sync(a_frag[i], a_half + (warp_row * 32 + i * TILE) * LD + kk, LD);
#pragma unroll
            for (unsigned int j = 0; j < 4; ++j) {
                nvcuda::wmma::load_matrix_sync(b_frag, b_half + (warp_col * 64 + j * TILE) * LD + kk, LD);
#pragma unroll
                for (unsigned int i = 0; i < 2; ++i)
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag, c_frag[i][j]);
            }
        }
        __syncthreads();
    }

    const bool full_tile = block_row + BM <= h && block_col + BN <= w && (w & 7) == 0;
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int row = block_row + warp_row * 32 + i * TILE;
            const unsigned int col = block_col + warp_col * 64 + j * TILE;
            if (full_tile) {
                nvcuda::wmma::store_matrix_sync(c + size_t(row) * w + col, c_frag[i][j], w, nvcuda::wmma::mem_row_major);
            } else {
                nvcuda::wmma::store_matrix_sync(c_float[warp], c_frag[i][j], TILE, nvcuda::wmma::mem_row_major);
                __syncwarp();
                for (unsigned int index = lane; index < TILE * TILE; index += 32) {
                    const unsigned int r = row + index / TILE;
                    const unsigned int n = col + index % TILE;
                    if (r < h && n < w)
                        c[size_t(r) * w + n] = c_float[warp][index];
                }
                __syncwarp();
            }
        }
    }
}

namespace cuda {
namespace {
template <typename Kernel>
void print_wmma_kernel_info(const char* name, Kernel kernel, int threads, const cudaDeviceProp& device)
{
    cudaFuncAttributes attributes;
    int blocks = 0;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attributes, kernel));
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, 0));
    std::printf("WMMA %s: threads/block=%d, registers/thread=%d, static shared/block=%zu B, local/thread=%zu B, binary=sm_%d, virtual=compute_%d\n",
        name, threads, attributes.numRegs, attributes.sharedSizeBytes, attributes.localSizeBytes,
        attributes.binaryVersion, attributes.ptxVersion);
    std::printf("WMMA %s theoretical occupancy limit: blocks/SM=%d, warps/SM=%d, threads=%.1f%%, preferred shared carveout=%d\n",
        name, blocks, blocks * threads / device.warpSize,
        100.0 * blocks * threads / device.maxThreadsPerMultiProcessor, attributes.preferredShmemCarveout);
}
}

void matrix_multiply_wmma_print_device_info()
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    int device_index = 0;
    int runtime_version = 0;
    int driver_version = 0;
    cudaDeviceProp device;
    CUDA_SAFE_CALL(cudaGetDevice(&device_index));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&device, device_index));
    CUDA_SAFE_CALL(cudaRuntimeGetVersion(&runtime_version));
    CUDA_SAFE_CALL(cudaDriverGetVersion(&driver_version));
    std::printf("WMMA device: %s, sm_%d%d, SMs=%d, nvcc=%d.%d, CUDART=%d, runtime=%d, driver=%d\n",
        device.name, device.major, device.minor, device.multiProcessorCount,
        __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, CUDART_VERSION, runtime_version, driver_version);
    std::printf("WMMA SM limits: registers=%d, shared=%zu B, max_threads=%d, warp=%d, L2=%d B\n",
        device.regsPerMultiprocessor, device.sharedMemPerMultiprocessor,
        device.maxThreadsPerMultiProcessor, device.warpSize, device.l2CacheSize);
    print_wmma_kernel_info("GEMM", ::matrix_multiply_wmma, 32 * MATRIX_WMMA_WARPS, device);
    print_wmma_kernel_info("packing", ::matrix_multiply_prepare_half, MATRIX_WMMA_PACK_THREADS, device);
    std::fflush(stdout);
}

void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::WorkSize& packingWorkSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c,
    gpu::gpu_mem_16u& a_packed, gpu::gpu_mem_16u& b_packed,
    unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const unsigned int packed_h = ((h + MATRIX_WMMA_BLOCK_M - 1) / MATRIX_WMMA_BLOCK_M) * MATRIX_WMMA_BLOCK_M;
    const unsigned int packed_w = ((w + MATRIX_WMMA_BLOCK_N - 1) / MATRIX_WMMA_BLOCK_N) * MATRIX_WMMA_BLOCK_N;
    const unsigned int packed_k = ((k + MATRIX_WMMA_BLOCK_K - 1) / MATRIX_WMMA_BLOCK_K) * MATRIX_WMMA_BLOCK_K;
    rassert(a_packed.number() >= size_t(packed_h) * packed_k, 1459342581, a_packed.number());
    rassert(b_packed.number() >= size_t(packed_w) * packed_k, 1459342582, b_packed.number());
    half* a_half = reinterpret_cast<half*>(a_packed.cuptr());
    half* b_half = reinterpret_cast<half*>(b_packed.cuptr());
    const unsigned int a_blocks = size_t(packed_h) * packed_k / (MATRIX_WMMA_PACK_THREADS * 4);
    ::matrix_multiply_prepare_half<<<packingWorkSize.cuGridSize(), packingWorkSize.cuBlockSize(), 0, stream>>>(
        a.cuptr(), b.cuptr(), a_half, b_half, w, h, k, packed_k, a_blocks, packed_w / 32);
    CUDA_CHECK_KERNEL_ASYNC(stream);
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        a_half, b_half, c.cuptr(), w, h, packed_k);
    CUDA_CHECK_KERNEL(stream);
}
}
