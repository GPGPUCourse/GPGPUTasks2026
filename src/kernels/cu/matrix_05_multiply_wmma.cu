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
    const unsigned int WARPS_PER_BLOCK = 8;
    __shared__ half tile_a[WARPS_PER_BLOCK][16 * 16];
    __shared__ half tile_b[WARPS_PER_BLOCK][16 * 16];

    const unsigned int lane = threadIdx.x;
    const unsigned int local_warp = threadIdx.y;
    const unsigned int warp_id = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x;
    const unsigned int tiles_x = w / 16;
    const unsigned int tile_i = (warp_id % tiles_x) * 16;
    const unsigned int tile_j = (warp_id / tiles_x) * 16;
    if (tile_j >= h)
        return;

    half* my_a = tile_a[local_warp];
    half* my_b = tile_b[local_warp];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (unsigned int t0 = 0; t0 < k; t0 += 16) {
        for (unsigned int e = lane; e < 16 * 16; e += 32) {
            const unsigned int r = e / 16;
            const unsigned int col = e % 16;
            my_a[e] = __float2half(a[(tile_j + r) * k + t0 + col]);
            my_b[e] = __float2half(b[(t0 + r) * w + tile_i + col]);
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, my_a, 16);
        wmma::load_matrix_sync(b_frag, my_b, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(c + tile_j * w + tile_i, c_frag, w, wmma::mem_row_major);
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
