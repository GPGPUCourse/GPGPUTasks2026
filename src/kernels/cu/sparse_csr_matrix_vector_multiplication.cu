#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* csr_row_offsets,
    const unsigned int* csr_columns,
    const unsigned int* csr_values,
    const unsigned int* vector_values,
          unsigned int* output_vector_values,
    unsigned int nrows)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows)
        return;

    unsigned long long acc = 0;
    const unsigned int row_from = csr_row_offsets[row];
    const unsigned int row_to = csr_row_offsets[row + 1];
    for (unsigned int i = row_from; i < row_to; ++i)
        acc += (unsigned long long) csr_values[i] * vector_values[csr_columns[i]];
    output_vector_values[row] = (unsigned int) acc;
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
        nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
