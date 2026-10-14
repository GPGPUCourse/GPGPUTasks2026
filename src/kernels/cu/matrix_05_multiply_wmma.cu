#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;



__global__ void float_to_half(
    const float* input,
    half* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}


__global__ void matrix_multiply_wmma(
        const half* A, // rows=h x cols=k
        const half* B, // rows=k x cols=w
        float* C, // rows=h x cols=w
        int W,
        int H,
        int K
    ) {
    int tile_row = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_col = blockIdx.x;

    wmma::fragment<
        wmma::matrix_a, 16, 16, 16,
        half, wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        half, wmma::row_major
    > b_frag;

    wmma::fragment<
        wmma::accumulator, 16, 16, 16,
        float
    > c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {

        const half* a_ptr =
            A + tile_row * 16 * K + k;

        const half* b_ptr =
            B + k * W + tile_col * 16;

        wmma::load_matrix_sync(
            a_frag, a_ptr, K
        );
        wmma::load_matrix_sync(
            b_frag, b_ptr, W
        );

        wmma::mma_sync(
            c_frag, a_frag, b_frag, c_frag
        );
    }

    float* c_ptr =
        C + tile_row * 16 * W + tile_col * 16;

    wmma::store_matrix_sync(
        c_ptr, c_frag, W, wmma::mem_row_major
    );
}




namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());

    cudaStream_t stream = context.cudaStream();

    gpu::gpu_mem_16u a_fp16; // rows=H x cols=K
    gpu::gpu_mem_16u b_fp16; // rows=K x cols=W

    a_fp16 = gpu::gpu_mem_16u(size_t(h) * k);
    b_fp16 = gpu::gpu_mem_16u(size_t(k) * w);

    gpu::WorkSize a_cast_worksize(GROUP_SIZE, h * k);
    ::float_to_half<<<a_cast_worksize.cuGridSize(), a_cast_worksize.cuBlockSize(), 0, stream>>>(a.cuptr(), (half*) a_fp16.cuptr(), h * k);

    gpu::WorkSize b_cast_worksize(GROUP_SIZE, k * w);
    ::float_to_half<<<b_cast_worksize.cuGridSize(), b_cast_worksize.cuBlockSize(), 0, stream>>>(b.cuptr(), (half*) b_fp16.cuptr(), k * w);

    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>((const half*) a_fp16.cuptr(), (const half*) b_fp16.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
