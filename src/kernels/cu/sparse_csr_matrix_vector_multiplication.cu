#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"
#include "libgpu/cuda/utils.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* row_offsets,
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* x,
    unsigned int* y,
    unsigned int nrows)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int row = tid / 32;
    if (row >= nrows)
        return;

    unsigned int begin = row_offsets[row];
    unsigned int end = row_offsets[row + 1];

    unsigned int sum = 0;
    for (unsigned int p = begin + lane; p < end; p += 32)
        sum += values[p] * x[columns[p]];
#pragma UNROLL
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0)
        y[row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& row_offsets,
    const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values,
    const gpu::gpu_mem_32u& x,
    gpu::gpu_mem_32u& y,
    unsigned int nrows)
{

    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA,
        34523543124312, context.type());

    cudaStream_t stream = context.cudaStream();

    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(),
        columns.cuptr(),
        values.cuptr(),
        x.cuptr(),
        y.cuptr(),
        nrows);

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
