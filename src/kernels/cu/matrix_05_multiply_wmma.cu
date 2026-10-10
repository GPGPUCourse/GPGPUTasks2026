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

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ __align__(32) half  sa[4][16][16];
    __shared__ __align__(32) half  sb[4][16][16];
    __shared__ __align__(32) float sc[4][16][16];

    const unsigned int warp = threadIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int tile_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned int tile_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (tile_row * 16 >= h || tile_col * 16 >= w)
        return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> facc;
    wmma::fill_fragment(facc, 0.0f);

    for (unsigned int t = 0; t < k; t += 16) {
        for (unsigned int e = lane; e < 256; e += 32) {
            const unsigned int r = e / 16;
            const unsigned int col = e % 16;
            const unsigned int aj = tile_row * 16 + r;
            const unsigned int ai = t + col;
            const unsigned int bj = t + r;
            const unsigned int bi = tile_col * 16 + col;
            sa[warp][r][col] = __float2half((aj < h && ai < k) ? a[aj * k + ai] : 0.0f);
            sb[warp][r][col] = __float2half((bj < k && bi < w) ? b[bj * w + bi] : 0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(fa, &sa[warp][0][0], 16);
        wmma::load_matrix_sync(fb, &sb[warp][0][0], 16);
        wmma::mma_sync(facc, fa, fb, facc);
        __syncwarp();
    }

    wmma::store_matrix_sync(&sc[warp][0][0], facc, 16, wmma::mem_row_major);
    __syncwarp();
    for (unsigned int e = lane; e < 256; e += 32) {
        const unsigned int r = e / 16;
        const unsigned int col = e % 16;
        const unsigned int cj = tile_row * 16 + r;
        const unsigned int ci = tile_col * 16 + col;
        if (cj < h && ci < w)
            c[cj * w + ci] = sc[warp][r][col];
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
