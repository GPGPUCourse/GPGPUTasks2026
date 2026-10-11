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

__global__ __launch_bounds__(32 * MATRIX_WMMA_WARPS, 2) void matrix_multiply_wmma(
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
    constexpr unsigned int STAGE_ELEMENTS = (BM + BN) * LD;
    constexpr unsigned int STAGE_VECTORS = STAGE_ELEMENTS / 8;
    constexpr unsigned int STRIP_ROWS = BM / 2;
    const unsigned int lane = threadIdx.x;
    const unsigned int warp = threadIdx.y;
    const unsigned int thread = warp * 32 + lane;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;

    __shared__ __align__(32) unsigned char shared_bytes[2 * STAGE_ELEMENTS * sizeof(half)];
    uint4* shared_vectors = reinterpret_cast<uint4*>(shared_bytes);
    float* output_tile = reinterpret_cast<float*>(shared_bytes);
    const unsigned int copy_row = (warp % 4) * 32 + 2 * ((lane / 4) % 4) + lane / 16;
    const unsigned int copy_vector = lane % 4;
    const unsigned int copy_offset = (warp < 4 ? 0 : BM * LD / 8) + copy_row * (LD / 8) + copy_vector;
    const unsigned int copy_origin = warp < 4 ? block_row : block_col;
    const half* copy_matrix = warp < 4 ? a : b;
    const uint4* input_vectors = reinterpret_cast<const uint4*>(copy_matrix)
        + size_t(copy_origin + copy_row) * (packed_k / 8) + copy_vector;

#pragma unroll
    for (unsigned int i = 0; i < 4; ++i)
        shared_vectors[copy_offset + i * LD] = input_vectors[size_t(i) * packed_k];
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2][4];
#pragma unroll
    for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < 4; ++j)
            nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    unsigned int stage = 0;
    for (unsigned int k0 = 0; k0 < packed_k; k0 += BK) {
        uint4 prefetched[4];
        const bool has_next = k0 + BK < packed_k;
        if (has_next) {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
                prefetched[i] = input_vectors[size_t(i) * packed_k + (k0 + BK) / 8];
        }
        half* a_half = reinterpret_cast<half*>(shared_bytes) + stage * STAGE_ELEMENTS;
        half* b_half = a_half + BM * LD;
#pragma unroll 1
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
        if (has_next) {
#pragma unroll
            for (unsigned int i = 0; i < 4; ++i)
                shared_vectors[(stage ^ 1) * STAGE_VECTORS + copy_offset + i * LD] = prefetched[i];
        }
        __syncthreads();
        stage ^= 1;
    }

    const bool full_tile = block_row + BM <= h && block_col + BN <= w && (w & 3) == 0;
#pragma unroll
    for (unsigned int strip = 0; strip < 2; ++strip) {
        if (warp / 4 == strip) {
#pragma unroll
            for (unsigned int i = 0; i < 2; ++i) {
#pragma unroll
                for (unsigned int j = 0; j < 4; ++j) {
                    const unsigned int row = (warp_row % 2) * 32 + i * TILE;
                    const unsigned int col = warp_col * 64 + j * TILE;
                    nvcuda::wmma::store_matrix_sync(output_tile + row * BN + col, c_frag[i][j], BN, nvcuda::wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
#pragma unroll 1
        for (unsigned int index = thread; index < STRIP_ROWS * BN / 4; index += 32 * WARPS) {
            const unsigned int row = block_row + strip * STRIP_ROWS + index / (BN / 4);
            const unsigned int col = block_col + 4 * (index % (BN / 4));
            const float4 value = reinterpret_cast<const float4*>(output_tile)[index];
            if (full_tile) {
                *reinterpret_cast<float4*>(c + size_t(row) * w + col) = value;
            } else if (row < h) {
                if (col < w)
                    c[size_t(row) * w + col] = value.x;
                if (col + 1 < w)
                    c[size_t(row) * w + col + 1] = value.y;
                if (col + 2 < w)
                    c[size_t(row) * w + col + 2] = value.z;
                if (col + 3 < w)
                    c[size_t(row) * w + col + 3] = value.w;
            }
        }
        __syncthreads();
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
    unsigned int w, unsigned int h, unsigned int k, bool print_timings)
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
    cudaEvent_t start = nullptr;
    cudaEvent_t after_packing = nullptr;
    cudaEvent_t after_gemm = nullptr;
    if (print_timings) {
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&after_packing));
        CUDA_SAFE_CALL(cudaEventCreate(&after_gemm));
        CUDA_SAFE_CALL(cudaEventRecord(start, stream));
    }
    ::matrix_multiply_prepare_half<<<packingWorkSize.cuGridSize(), packingWorkSize.cuBlockSize(), 0, stream>>>(
        a.cuptr(), b.cuptr(), a_half, b_half, w, h, k, packed_k, a_blocks, packed_w / 32);
    CUDA_CHECK_KERNEL_ASYNC(stream);
    if (print_timings)
        CUDA_SAFE_CALL(cudaEventRecord(after_packing, stream));
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        a_half, b_half, c.cuptr(), w, h, packed_k);
    if (print_timings) {
        CUDA_SAFE_CALL(cudaEventRecord(after_gemm, stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(after_gemm));
        float packing_ms = 0.0f;
        float gemm_ms = 0.0f;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&packing_ms, start, after_packing));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&gemm_ms, after_packing, after_gemm));
        std::printf("WMMA stage timing after warmup: packing=%.6f ms, GEMM=%.6f ms, total=%.6f ms\n",
            packing_ms, gemm_ms, packing_ms + gemm_ms);
        std::fflush(stdout);
        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(after_packing));
        CUDA_SAFE_CALL(cudaEventDestroy(after_gemm));
    }
    CUDA_CHECK_KERNEL(stream);
}
}
