#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* offsets,
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* input,
          unsigned int* output,
    unsigned int nrows)
{
    const unsigned int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (row >= nrows) {
        return;
    }

    unsigned int sum = 0;
    for (unsigned int i = offsets[row] + lane; i < offsets[row + 1]; i += 32) {
        sum += values[i] * input[columns[i]];
    }
    for (unsigned int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &offsets, const gpu::gpu_mem_32u &columns, const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    if (nrows == 0) {
        return;
    }
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        offsets.cuptr(), columns.cuptr(), values.cuptr(),
        input.cuptr(), output.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
