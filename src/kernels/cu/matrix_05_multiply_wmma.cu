#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__global__ void matrix_multiply_wmma(
                       const float* a,
                       const float* b,
                             float* c,
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int tileColIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    unsigned int tileRowIdx =  blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tileCol = tileColIdx * WMMA_N;
    unsigned int tileRow = tileRowIdx * WMMA_M;

    if (tileRow >= h || tileCol >= w) {
        return;
    }

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>  a_frag;
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>   b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (unsigned int k_step = 0; k_step < k; k_step += WMMA_K) {
        const float* a_ptr = a + (size_t)tileRow * k + k_step;
        const float* b_ptr = b + (size_t)k_step  * w + tileCol;

        wmma::load_matrix_sync(a_frag, a_ptr, k);
        wmma::load_matrix_sync(b_frag, b_ptr, w);

        // Округление float -> tf32
        for (int i = 0; i < a_frag.num_elements; i++) a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
        for (int i = 0; i < b_frag.num_elements; i++) b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* c_ptr = c + (size_t)tileRow * w + tileCol;
    wmma::store_matrix_sync(c_ptr, acc_frag, w, wmma::mem_row_major);
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
