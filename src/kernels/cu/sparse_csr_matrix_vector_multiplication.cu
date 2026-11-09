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
                                                              unsigned int* output_vector_values,
                                                        unsigned int nrows)
{
    const unsigned int local_row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const unsigned int line = threadIdx.x & 31;

    if (local_row >= nrows)
        return;

    const unsigned int start = csr_row_offsets[local_row];
    const unsigned int end = csr_row_offsets[local_row + 1];

    unsigned int sum = 0;
    for (unsigned int i = start + line; i < end; i += 32) {
        const unsigned int col = csr_columns[i];
        sum += csr_values[i] * vector_values[col];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (line == 0)
        output_vector_values[local_row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
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
