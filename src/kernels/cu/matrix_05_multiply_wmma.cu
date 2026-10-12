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

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[2];

    uint4 pre_a[2], pre_b[2];

    auto load_global = [&](int k_curr) {
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            int chunk = tid + rep * 128;

            int r_a = chunk / 4;
            int c_a = (chunk % 4) * 8;
            int g_row_a = block_row + r_a;
            int g_col_a = k_curr + c_a;
            pre_a[rep] = (g_row_a < h && g_col_a + 7 < k)
                ? *reinterpret_cast<const uint4*>(&a[(size_t)g_row_a * k + g_col_a])
                : make_uint4(0, 0, 0, 0);

            int r_b = chunk / 8;
            int c_b = (chunk % 8) * 8;
            int g_row_b = k_curr + r_b;
            int g_col_b = block_col + c_b;
            pre_b[rep] = (g_row_b < k && g_col_b + 7 < w)
                ? *reinterpret_cast<const uint4*>(&b[(size_t)g_row_b * w + g_col_b])
                : make_uint4(0, 0, 0, 0);
        }
    };

    auto store_shared = [&](int stage) {
        #pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            int chunk = tid + rep * 128;
            int r_a = chunk / 4;
            int c_a = (chunk % 4) * 8;
            *reinterpret_cast<uint4*>(&local_a[stage][r_a][c_a]) = pre_a[rep];

            int r_b = chunk / 8;
            int c_b = (chunk % 8) * 8;
            *reinterpret_cast<uint4*>(&local_b[stage][r_b][c_b]) = pre_b[rep];
        }
    };

    load_global(0);
    store_shared(0);
    __syncthreads();

    int stage = 0;

    for (unsigned int k_step = 0; k_step < k; k_step += BK) {
        unsigned int next_k = k_step + BK;

        if (next_k < k) {
            load_global(next_k);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bf[j], &local_b[stage][kk][warp_col + j * 16], BN + PAD);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(af, &local_a[stage][warp_row + i * 16][kk], BK + PAD);
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(c_frag[i][j], af, bf[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();

        if (next_k < k) {
            store_shared(stage ^ 1);
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