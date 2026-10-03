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

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void fp32_to_fp16(float *in, half *out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void matrix_multiply_wmma(
                       const half* a, // rows=h x cols=k
                       const half* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = k;
    int ldb = w;
    int ldc = w;

    // Tile using a 2D grid
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
 
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < h && aCol < k && bRow < k && bCol < w) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < h && cCol < w) {
        // Store the output
        wmma::store_matrix_sync(c + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

namespace cuda {
void matrix_multiply_wmma(
            const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a,
            const gpu::gpu_mem_16f &at,
            const gpu::gpu_mem_32f &b,
            const gpu::gpu_mem_16f &bt,
            gpu::gpu_mem_32f &c,
            unsigned int w,
            unsigned int h,
            unsigned int k
) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    auto work_size_at = WorkSize(GROUP_SIZE, h * k);
    auto work_size_bt = WorkSize(GROUP_SIZE, k * w);
    ::fp32_to_fp16<<<work_size_at.cuGridSize(), work_size_at.cuBlockSize(), 0, stream>>>(a.cuptr(), at.cuptr(), h * k);
    CUDA_CHECK_KERNEL(stream);
    ::fp32_to_fp16<<<work_size_bt.cuGridSize(), work_size_bt.cuBlockSize(), 0, stream>>>(b.cuptr(), bt.cuptr(), k * w);
    CUDA_CHECK_KERNEL(stream);

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (w + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (h + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    ::matrix_multiply_wmma<<<gridDim, blockDim, 0, stream>>>(at.cuptr(), bt.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
