#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__launch_bounds__(256, 2)
__global__ void matrix_multiply_wmma(
                       const float* a,
                       const float* b,
                             float* c,
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int BM = 128;
    const unsigned int BN = 128;
    const unsigned int BK = 16;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;

    const unsigned int warp_row = (warp_id / 4) * 64;
    const unsigned int warp_col = (warp_id % 4) * 32;

    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;

    __shared__ alignas(16) half sh_a[2][128][16 + 8];
    __shared__ alignas(16) half sh_b[2][16][128 + 8];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    const unsigned int a_r0 = tid / 4;
    const unsigned int a_c0 = (tid % 4) * 4;
    const unsigned int a_r1 = (tid + 256) / 4;
    const unsigned int a_c1 = ((tid + 256) % 4) * 4;

    const unsigned int b_r0 = tid / 32;
    const unsigned int b_c0 = (tid % 32) * 4;
    const unsigned int b_r1 = (tid + 256) / 32;
    const unsigned int b_c1 = ((tid + 256) % 32) * 4;

    float4 fa0 = *reinterpret_cast<const float4*>(&a[(block_row + a_r0) * k + a_c0]);
    float4 fa1 = *reinterpret_cast<const float4*>(&a[(block_row + a_r1) * k + a_c1]);

    half2 ha0_0 = __float22half2_rn(make_float2(fa0.x, fa0.y));
    half2 ha0_1 = __float22half2_rn(make_float2(fa0.z, fa0.w));
    half2 ha1_0 = __float22half2_rn(make_float2(fa1.x, fa1.y));
    half2 ha1_1 = __float22half2_rn(make_float2(fa1.z, fa1.w));

    *reinterpret_cast<half2*>(&sh_a[0][a_r0][a_c0]) = ha0_0;
    *reinterpret_cast<half2*>(&sh_a[0][a_r0][a_c0 + 2]) = ha0_1;
    *reinterpret_cast<half2*>(&sh_a[0][a_r1][a_c1]) = ha1_0;
    *reinterpret_cast<half2*>(&sh_a[0][a_r1][a_c1 + 2]) = ha1_1;

    float4 fb0 = *reinterpret_cast<const float4*>(&b[b_r0 * w + block_col + b_c0]);
    float4 fb1 = *reinterpret_cast<const float4*>(&b[b_r1 * w + block_col + b_c1]);

    half2 hb0_0 = __float22half2_rn(make_float2(fb0.x, fb0.y));
    half2 hb0_1 = __float22half2_rn(make_float2(fb0.z, fb0.w));
    half2 hb1_0 = __float22half2_rn(make_float2(fb1.x, fb1.y));
    half2 hb1_1 = __float22half2_rn(make_float2(fb1.z, fb1.w));

    *reinterpret_cast<half2*>(&sh_b[0][b_r0][b_c0]) = hb0_0;
    *reinterpret_cast<half2*>(&sh_b[0][b_r0][b_c0 + 2]) = hb0_1;
    *reinterpret_cast<half2*>(&sh_b[0][b_r1][b_c1]) = hb1_0;
    *reinterpret_cast<half2*>(&sh_b[0][b_r1][b_c1 + 2]) = hb1_1;

    __syncthreads();

    int stage = 0;
    for (unsigned int tile_k = 0; tile_k < k; tile_k += BK) {
        const unsigned int next_k = tile_k + BK;
        const int next_stage = 1 - stage;

        half2 next_ha0_0, next_ha0_1, next_ha1_0, next_ha1_1;
        half2 next_hb0_0, next_hb0_1, next_hb1_0, next_hb1_1;

        if (next_k < k) {
            float4 nfa0 = *reinterpret_cast<const float4*>(&a[(block_row + a_r0) * k + next_k + a_c0]);
            float4 nfa1 = *reinterpret_cast<const float4*>(&a[(block_row + a_r1) * k + next_k + a_c1]);

            next_ha0_0 = __float22half2_rn(make_float2(nfa0.x, nfa0.y));
            next_ha0_1 = __float22half2_rn(make_float2(nfa0.z, nfa0.w));
            next_ha1_0 = __float22half2_rn(make_float2(nfa1.x, nfa1.y));
            next_ha1_1 = __float22half2_rn(make_float2(nfa1.z, nfa1.w));

            float4 nfb0 = *reinterpret_cast<const float4*>(&b[(next_k + b_r0) * w + block_col + b_c0]);
            float4 nfb1 = *reinterpret_cast<const float4*>(&b[(next_k + b_r1) * w + block_col + b_c1]);

            next_hb0_0 = __float22half2_rn(make_float2(nfb0.x, nfb0.y));
            next_hb0_1 = __float22half2_rn(make_float2(nfb0.z, nfb0.w));
            next_hb1_0 = __float22half2_rn(make_float2(nfb1.x, nfb1.y));
            next_hb1_1 = __float22half2_rn(make_float2(nfb1.z, nfb1.w));
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            wmma::load_matrix_sync(a_frag[i], &sh_a[stage][warp_row + i * 16][0], 16 + 8);
        }

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::load_matrix_sync(b_frag[j], &sh_b[stage][0][warp_col + j * 16], 128 + 8);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        if (next_k < k) {
            *reinterpret_cast<half2*>(&sh_a[next_stage][a_r0][a_c0]) = next_ha0_0;
            *reinterpret_cast<half2*>(&sh_a[next_stage][a_r0][a_c0 + 2]) = next_ha0_1;
            *reinterpret_cast<half2*>(&sh_a[next_stage][a_r1][a_c1]) = next_ha1_0;
            *reinterpret_cast<half2*>(&sh_a[next_stage][a_r1][a_c1 + 2]) = next_ha1_1;

            *reinterpret_cast<half2*>(&sh_b[next_stage][b_r0][b_c0]) = next_hb0_0;
            *reinterpret_cast<half2*>(&sh_b[next_stage][b_r0][b_c0 + 2]) = next_hb0_1;
            *reinterpret_cast<half2*>(&sh_b[next_stage][b_r1][b_c1]) = next_hb1_0;
            *reinterpret_cast<half2*>(&sh_b[next_stage][b_r1][b_c1 + 2]) = next_hb1_1;
        }

        __syncthreads();
        stage = next_stage;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const unsigned int r = block_row + warp_row + i * 16;
            const unsigned int c_idx = block_col + warp_col + j * 16;
            if (r < h && c_idx < w) {
                wmma::store_matrix_sync(c + r * w + c_idx, c_frag[i][j], w, wmma::mem_row_major);
            }
        }
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    dim3 block(256);
    dim3 grid((w + 127) / 128, (h + 127) / 128);
    ::matrix_multiply_wmma<<<grid, block, 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
