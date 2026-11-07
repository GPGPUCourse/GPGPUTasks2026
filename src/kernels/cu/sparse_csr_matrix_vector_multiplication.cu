#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication( // TODO input/output buffers
    const unsigned int* row_ptr,
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* x,
    unsigned int* y,
    unsigned int nrows)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int row = tid / 32;
    unsigned int lane = tid % 32;

    if (row >= nrows)
        return;

    unsigned int sum = 0;

    unsigned int start = row_ptr[row];
    unsigned int end = row_ptr[row + 1];

    for (unsigned int i = start + lane; i < end; i += 32) {
        sum += values[i] * x[columns[i]];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& row_ptr,
    const gpu::gpu_mem_32u& columns,
    const gpu::gpu_mem_32u& values,
    const gpu::gpu_mem_32u& x,
    gpu::gpu_mem_32u& y,
    unsigned int nrows)
// TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_ptr.cuptr(),
        columns.cuptr(),
        values.cuptr(),
        x.cuptr(),
        y.cuptr(),
        nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
