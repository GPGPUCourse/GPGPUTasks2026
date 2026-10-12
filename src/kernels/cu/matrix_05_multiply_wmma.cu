#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__device__ __forceinline__ uint2 pack_wmma_vector(float4 value)
{
    const half2 low = __float22half2_rn(make_float2(value.x, value.y));
    const half2 high = __float22half2_rn(make_float2(value.z, value.w));
    return make_uint2(reinterpret_cast<const unsigned int&>(low), reinterpret_cast<const unsigned int&>(high));
}

__device__ __forceinline__ void store_wmma_vector(half* destination, float4 value)
{
    *reinterpret_cast<uint2*>(destination) = pack_wmma_vector(value);
}

__launch_bounds__(256, 2)
__global__ void matrix_multiply_wmma(
                       const float* __restrict__ a,
                       const float* __restrict__ b,
                             float* __restrict__ c,
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int BM = 128;
    const unsigned int BN = 128;
    const unsigned int BK = 16;
    const unsigned int LOADS = BM * BK / (256 * 4);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;

    const unsigned int warp_row = (warp_id / 4) * 64;
    const unsigned int warp_col = (warp_id % 4) * 32;

    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;

    __shared__ alignas(32) half sh_a[2][BM][BK + 8];
    __shared__ alignas(32) half sh_b[2][BK][BN + 8];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    #pragma unroll
    for (unsigned int i = 0; i < LOADS; ++i) {
        const unsigned int index = (tid + i * 256) * 4;
        const unsigned int ar = index / BK;
        const unsigned int ac = index % BK;
        const unsigned int br = index / BN;
        const unsigned int bc = index % BN;
        store_wmma_vector(&sh_a[0][ar][ac], *reinterpret_cast<const float4*>(a + (block_row + ar) * k + ac));
        store_wmma_vector(&sh_b[0][br][bc], *reinterpret_cast<const float4*>(b + br * w + block_col + bc));
    }

    __syncthreads();

    int stage = 0;
    for (unsigned int tile_k = 0; tile_k < k; tile_k += BK) {
        const unsigned int next_k = tile_k + BK;
        const int next_stage = 1 - stage;

        uint2 next_a[LOADS];
        uint2 next_b[LOADS];

        if (next_k < k) {
            #pragma unroll
            for (unsigned int i = 0; i < LOADS; ++i) {
                const unsigned int index = (tid + i * 256) * 4;
                next_a[i] = pack_wmma_vector(*reinterpret_cast<const float4*>(a + (block_row + index / BK) * k + next_k + index % BK));
                next_b[i] = pack_wmma_vector(*reinterpret_cast<const float4*>(b + (next_k + index / BN) * w + block_col + index % BN));
            }
        }

        #pragma unroll
        for (unsigned int inner = 0; inner < BK; inner += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::load_matrix_sync(a_frag[i], &sh_a[stage][warp_row + i * 16][inner], BK + 8);
            }
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(b_frag[j], &sh_b[stage][inner][warp_col + j * 16], BN + 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        if (next_k < k) {
            #pragma unroll
            for (unsigned int i = 0; i < LOADS; ++i) {
                const unsigned int index = (tid + i * 256) * 4;
                *reinterpret_cast<uint2*>(&sh_a[next_stage][index / BK][index % BK]) = next_a[i];
                *reinterpret_cast<uint2*>(&sh_b[next_stage][index / BN][index % BN]) = next_b[i];
            }
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
            wmma::store_matrix_sync(c + r * w + c_idx, c_frag[i][j], w, wmma::mem_row_major);
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
    dim3 grid(w / 128, h / 128);
    ::matrix_multiply_wmma<<<grid, block, 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
