#include <cuda_fp16.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

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
    // TODO 020 Это добровольное задание за супер-пупер-баллы престижа сверх нормы
    int corner_i = blockIdx.y * 16, corner_j = blockIdx.x * 16;
    int idx = threadIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> tmp_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> tmp_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

    wmma::fill_fragment(acc, 0);

    __shared__ __align__(32) half tile_a[16 * 16], tile_b[16 * 16];
    __shared__ __align__(32) float tile_c[16 * 16];

    for (int base = 0; base < k; base += 16) {
        for (int i = idx; i < 256; i += 32) {
            int row = i / 16, col = i % 16;

            tile_a[i] = corner_i + row < h && base + col < k ? __float2half(a[(corner_i + row) * k + base + col]) : half(0);
            tile_b[i] = corner_j + col < w && base + row < k ? __float2half(b[(base + row) * w + corner_j + col]) : half(0);
        }

        __syncwarp();

        wmma::load_matrix_sync(tmp_a, tile_a, 16);
        wmma::load_matrix_sync(tmp_b, tile_b, 16);
        wmma::mma_sync(acc, tmp_a, tmp_b, acc);

        __syncwarp();
    }

    wmma::store_matrix_sync(tile_c, acc, 16, wmma::mem_row_major);

    __syncwarp();

    for (int i = idx; i < 256; i += 32) {
        int row = corner_i +i / 16, col = corner_j + i % 16;

        if (row < h && col < w) {
            c[row * w + col] = tile_c[i];
        }
    }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
