#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// TODO input/output buffers
__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* row_offsets, const unsigned int* columns,
    const unsigned int* values, const unsigned int* vector_values,
    unsigned int* output, unsigned int nrows)
{
    // TODO
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int row = index / warpSize;
    const unsigned int lane = threadIdx.x % warpSize;
    if (row >= nrows)
        return;

    unsigned int sum = 0;
    for (size_t i = size_t(row_offsets[row]) + lane; i < row_offsets[row + 1]; i += warpSize)
        sum += values[i] * vector_values[columns[i]];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[row] = sum;
}

namespace cuda {
// TODO input/output buffers
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_offsets, const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &vector_values,
    gpu::gpu_mem_32u &output, unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        // TODO input/output buffers
        // input_buffer.cuptr(),
        row_offsets.cuptr(), columns.cuptr(), values.cuptr(), vector_values.cuptr(), output.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
