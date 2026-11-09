#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
        unsigned int *row_offsets,
        unsigned int *cols,
        unsigned int *values,
        unsigned int *vector_values,
        unsigned int *out,
        unsigned int nrows) // TODO input/output buffers
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nrows)
        return;

    unsigned int sum = 0;
    for (unsigned int i = row_offsets[index]; i < row_offsets[index + 1]; ++i)
    {
        sum += values[i] * vector_values[cols[i]];
    }

    out[index] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &row_offsets, const gpu::gpu_mem_32u &cols, gpu::gpu_mem_32u &values,
                                             gpu::gpu_mem_32u &vector_values, gpu::gpu_mem_32u &out, unsigned int nrows) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(),
        cols.cuptr(),
        values.cuptr(),
        vector_values.cuptr(),
        out.cuptr(),
        nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
