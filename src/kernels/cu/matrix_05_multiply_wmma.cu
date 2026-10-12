#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <cuda_fp16.h>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* __restrict__ a, // rows=h x cols=k
                       const float* __restrict__ b, // rows=k x cols=w
                             float* __restrict__ c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int AP = BK + 8;
    constexpr int BP = BN + 8;

    __shared__ __align__(16) half tileA[BM][AP];
    __shared__ __align__(16) half tileB[BK][BP];

    const unsigned int tid = threadIdx.y * 16 + threadIdx.x;
    const unsigned int warp = tid / 32;

    const unsigned int row0 = blockIdx.y * BM;
    const unsigned int col0 = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        wmma::fill_fragment(acc[j], 0.0f);
    }

    for (unsigned int k0 = 0; k0 < k; k0 += BK) {
        __syncthreads();

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const unsigned int p = 4 * (tid + step * 256);

            const unsigned int ar = p / BK;
            const unsigned int ac = p % BK;

            const unsigned int br = p / BN;
            const unsigned int bc = p % BN;

            const float4 av = *reinterpret_cast<const float4*>(a + (size_t)(row0 + ar) * k + k0 + ac);
            const float4 bv = *reinterpret_cast<const float4*>(b + (size_t)(k0 + br) * w + col0 + bc);

            *reinterpret_cast<half2*>(&tileA[ar][ac]) = __floats2half2_rn(av.x, av.y);
            *reinterpret_cast<half2*>(&tileA[ar][ac + 2]) = __floats2half2_rn(av.z, av.w);
            *reinterpret_cast<half2*>(&tileB[br][bc]) = __floats2half2_rn(bv.x, bv.y);
            *reinterpret_cast<half2*>(&tileB[br][bc + 2]) = __floats2half2_rn(bv.z, bv.w);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::load_matrix_sync(af, &tileA[warp * 16][kk], AP);

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
                wmma::load_matrix_sync(bf, &tileB[kk][j * 16], BP);
                wmma::mma_sync(acc[j], af, bf, acc[j]);
            }
        }
    }

    const unsigned int rowC = row0 + warp * 16;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        wmma::store_matrix_sync(c + (size_t)rowC * w + col0 + j * 16, acc[j], w, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
