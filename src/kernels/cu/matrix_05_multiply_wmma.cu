#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define BM 64
#define BN 64
#define BK 32
#define PAD 8

__global__ void f32_to_f16_kernel(const float* __restrict__ in, half* __restrict__ out, size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(in + i);
        reinterpret_cast<half2*>(out + i)[0] = __floats2half2_rn(v.x, v.y);
        reinterpret_cast<half2*>(out + i)[1] = __floats2half2_rn(v.z, v.w);
    }
}

__global__ void __launch_bounds__(128, 4) matrix_multiply_wmma(
                       const half* __restrict__ a,  // rows=h x cols=k
                       const half* __restrict__ b,  // rows=k x cols=w
                             float* __restrict__ c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ half local_a[2][BM][BK + PAD];
    __shared__ half local_b[2][BK][BN + PAD];

    unsigned int block_row = blockIdx.y * BM;
    unsigned int block_col = blockIdx.x * BN;

    unsigned int tid = threadIdx.y * 32 + threadIdx.x;
    unsigned int warp_id = threadIdx.y;

    unsigned int warp_row = (warp_id / 2) * 32;
    unsigned int warp_col = (warp_id % 2) * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];

    int smem_ar[2], smem_ac[2];
    int smem_br[2], smem_bc[2];

    #pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        int chunk = tid + rep * 128;
        smem_ar[rep] = chunk / 4;
        smem_ac[rep] = (chunk % 4) * 8;

        smem_br[rep] = chunk / 8;
        smem_bc[rep] = (chunk % 8) * 8;
    }


    const half* ptr_a0 = a + (size_t)(block_row + smem_ar[0]) * k + smem_ac[0];
    const half* ptr_a1 = a + (size_t)(block_row + smem_ar[1]) * k + smem_ac[1];
    const half* ptr_b0 = b + (size_t)(smem_br[0]) * w + (block_col + smem_bc[0]);
    const half* ptr_b1 = b + (size_t)(smem_br[1]) * w + (block_col + smem_bc[1]);

    uint4 pre_a[2], pre_b[2];


    pre_a[0] = *reinterpret_cast<const uint4*>(ptr_a0);
    pre_a[1] = *reinterpret_cast<const uint4*>(ptr_a1);
    pre_b[0] = *reinterpret_cast<const uint4*>(ptr_b0);
    pre_b[1] = *reinterpret_cast<const uint4*>(ptr_b1);

    #pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        *reinterpret_cast<uint4*>(&local_a[0][smem_ar[rep]][smem_ac[rep]]) = pre_a[rep];
        *reinterpret_cast<uint4*>(&local_b[0][smem_br[rep]][smem_bc[rep]]) = pre_b[rep];
    }
    __syncthreads();

    int stage = 0;


    for (unsigned int k_step = 0; k_step < k; k_step += BK) {
        unsigned int next_k = k_step + BK;


        if (next_k < k) {
            pre_a[0] = *reinterpret_cast<const uint4*>(ptr_a0 + next_k);
            pre_a[1] = *reinterpret_cast<const uint4*>(ptr_a1 + next_k);
            pre_b[0] = *reinterpret_cast<const uint4*>(ptr_b0 + (size_t)next_k * w);
            pre_b[1] = *reinterpret_cast<const uint4*>(ptr_b1 + (size_t)next_k * w);
        }


        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bf[j], &local_b[stage][kk][warp_col + j * 16], BN + PAD);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(af[i], &local_a[stage][warp_row + i * 16][kk], BK + PAD);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(c_frag[i][j], af[i], bf[j], c_frag[i][j]);
                }
            }
        }


        if (next_k < k) {
            #pragma unroll
            for (int rep = 0; rep < 2; ++rep) {
                *reinterpret_cast<uint4*>(&local_a[stage ^ 1][smem_ar[rep]][smem_ac[rep]]) = pre_a[rep];
                *reinterpret_cast<uint4*>(&local_b[stage ^ 1][smem_br[rep]][smem_bc[rep]]) = pre_b[rep];
            }
        }


        __syncthreads();
        stage ^= 1;
    }


    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            unsigned int out_row = block_row + warp_row + i * 16;
            unsigned int out_col = block_col + warp_col + j * 16;
            if (out_row < h && out_col < w) {
                wmma::store_matrix_sync(c + (size_t)out_row * w + out_col, c_frag[i][j], w, wmma::mem_row_major);
            }
        }
    }
}

namespace cuda {

struct Fp16Cache {
    const float* src = nullptr;
    size_t count = 0;
    half* gpu_data = nullptr;
    size_t capacity = 0;
};

void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    static Fp16Cache cache_a, cache_b;
    const size_t na = (size_t)h * k;
    const size_t nb = (size_t)k * w;

    if (cache_a.capacity < na) {
        if (cache_a.gpu_data) cudaFree(cache_a.gpu_data);
        cudaMalloc(&cache_a.gpu_data, na * sizeof(half));
        cache_a.capacity = na;
        cache_a.src = nullptr;
    }
    if (cache_a.src != a.cuptr() || cache_a.count != na) {
        int threads = 256;
        int blocks = (int)((na / 4 + threads - 1) / threads);
        f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(a.cuptr(), cache_a.gpu_data, na);
        cache_a.src = a.cuptr();
        cache_a.count = na;
    }

    if (cache_b.capacity < nb) {
        if (cache_b.gpu_data) cudaFree(cache_b.gpu_data);
        cudaMalloc(&cache_b.gpu_data, nb * sizeof(half));
        cache_b.capacity = nb;
        cache_b.src = nullptr;
    }
    if (cache_b.src != b.cuptr() || cache_b.count != nb) {
        int threads = 256;
        int blocks = (int)((nb / 4 + threads - 1) / threads);
        f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(b.cuptr(), cache_b.gpu_data, nb);
        cache_b.src = b.cuptr();
        cache_b.count = nb;
    }

    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        cache_a.gpu_data, cache_b.gpu_data, c.cuptr(), w, h, k
    );

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda