#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(const unsigned int* csr_row_offsets,
                                                        const unsigned int* csr_columns,
                                                        const unsigned int* csr_values,
                                                        const unsigned int* vector_values,
                                                              unsigned int* result,
                                                              unsigned int  nrows)
{
    const unsigned int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;

    // row одинаков для всего варпа, поэтому варп выходит целиком и __shfl_down_sync ниже безопасен
    if (row >= nrows)
        return;

    unsigned int sum = 0;
    for (unsigned int i = csr_row_offsets[row] + lane; i < csr_row_offsets[row + 1]; i += WARP_SIZE) {
        sum += csr_values[i] * vector_values[csr_columns[i]];
    }

    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        result[row] = sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::gpu_mem_32u &csr_row_offsets, const gpu::gpu_mem_32u &csr_columns, const gpu::gpu_mem_32u &csr_values,
                                             const gpu::gpu_mem_32u &vector_values, gpu::gpu_mem_32u &result, unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const gpu::WorkSize workSize(GROUP_SIZE, (size_t) nrows * WARP_SIZE);
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(), csr_columns.cuptr(), csr_values.cuptr(), vector_values.cuptr(), result.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
