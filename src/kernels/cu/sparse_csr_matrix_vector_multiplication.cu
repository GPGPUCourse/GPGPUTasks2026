#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define WARP_SIZE 32

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* __restrict__ csr_row_offsets,
    const unsigned int* __restrict__ csr_columns,
    const unsigned int* __restrict__ csr_values,
    const unsigned int* __restrict__ vector_values,
          unsigned int* __restrict__ output_vector_values,
          unsigned int               nrows)
{
    const unsigned int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row  = global_thread_id / WARP_SIZE;
    const unsigned int lane = threadIdx.x & (WARP_SIZE - 1);

    if (row >= nrows)
        return;

    const unsigned int row_from = __ldg(&csr_row_offsets[row]);
    const unsigned int row_to   = __ldg(&csr_row_offsets[row + 1]);

    unsigned int sum = 0;
    unsigned int i = row_from + lane;

    for (; i + WARP_SIZE < row_to; i += 2 * WARP_SIZE) {
        const unsigned int col0 = __ldg(&csr_columns[i]);
        const unsigned int col1 = __ldg(&csr_columns[i + WARP_SIZE]);
        const unsigned int val0 = __ldg(&csr_values[i]);
        const unsigned int val1 = __ldg(&csr_values[i + WARP_SIZE]);

        sum += val0 * __ldg(&vector_values[col0]) + val1 * __ldg(&vector_values[col1]);
    }

    if (i < row_to) {
        const unsigned int col0 = __ldg(&csr_columns[i]);
        const unsigned int val0 = __ldg(&csr_values[i]);
        sum += val0 * __ldg(&vector_values[col0]);
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output_vector_values[row] = sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &csr_row_offsets,
    const gpu::gpu_mem_32u &csr_columns,
    const gpu::gpu_mem_32u &csr_values,
    const gpu::gpu_mem_32u &vector_values,
          gpu::gpu_mem_32u &output_vector_values,
          unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(),
        csr_columns.cuptr(),
        csr_values.cuptr(),
        vector_values.cuptr(),
        output_vector_values.cuptr(),
        nrows
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda