#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
using namespace nvcuda;
#define PAD 8

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ half local_a[64][16 + PAD];
    __shared__ half local_b[16][64 + PAD];

    unsigned int block_row = blockIdx.y * 64;
    unsigned int block_col = blockIdx.x * 64;

    unsigned int tid = threadIdx.y * 32 + threadIdx.x; // 0..511
    unsigned int warp_id = threadIdx.y;                // 0..15

    unsigned int warp_row = (warp_id / 4) * 16;
    unsigned int warp_col = (warp_id % 4) * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (unsigned int i = 0; i < k; i += 16) {


        if (tid < 256) {
            int row = tid / 4;
            int col_4 = (tid % 4) * 4;
            int g_row = block_row + row;
            int g_col = i + col_4;

            float4 val = (g_row < h && g_col + 3 < k)
                ? *reinterpret_cast<const float4*>(&a[g_row * k + g_col])
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            *reinterpret_cast<half2*>(&local_a[row][col_4])     = __floats2half2_rn(val.x, val.y);
            *reinterpret_cast<half2*>(&local_a[row][col_4 + 2]) = __floats2half2_rn(val.z, val.w);
        }


        if (tid >= 256) {
            int local_tid = tid - 256;
            int row = local_tid / 16;
            int col_4 = (local_tid % 16) * 4;
            int g_row = i + row;
            int g_col = block_col + col_4;

            float4 val = (g_row < k && g_col + 3 < w)
                ? *reinterpret_cast<const float4*>(&b[g_row * w + g_col])
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            *reinterpret_cast<half2*>(&local_b[row][col_4])     = __floats2half2_rn(val.x, val.y);
            *reinterpret_cast<half2*>(&local_b[row][col_4 + 2]) = __floats2half2_rn(val.z, val.w);
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, &local_a[warp_row][0], 16 + PAD);
        wmma::load_matrix_sync(b_frag, &local_b[0][warp_col], 64 + PAD);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    unsigned int out_row = block_row + warp_row;
    unsigned int out_col = block_col + warp_col;

    if (out_row < h && out_col < w) {
        wmma::store_matrix_sync(c + out_row * w + out_col, c_frag, w, wmma::mem_row_major);
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
