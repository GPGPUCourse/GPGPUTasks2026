#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* __restrict__ row_offsets,
    const unsigned int* __restrict__ columns,
    const unsigned int* __restrict__ values,
    const unsigned int* __restrict__ x,
    unsigned int* __restrict__ y,
    unsigned int nrows)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= nrows) 
        return;
    
    const unsigned int row_begin = row_offsets[row];
    const unsigned int row_end = row_offsets[row + 1];
    unsigned int sum = 0;
    unsigned int i = row_begin;
    while (i < row_end) {
        const unsigned int col = columns[i];
        sum += values[i] * x[col];
        ++i;
    }

    y[row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_offsets,
    const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &x,
    gpu::gpu_mem_32u &y,
    unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(), columns.cuptr(), values.cuptr(), x.cuptr(), y.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
