#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication(
    unsigned int nrows,
    unsigned int* csr_row_offsets,
    unsigned int* csr_columns,
    unsigned int* csr_values,
    unsigned int* vector_values,
    unsigned int* output_values)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nrows) {
        return;
    }
    unsigned int accumulator = 0;
    for (int i = csr_row_offsets[index]; i < csr_row_offsets[index + 1]; i++) {
        accumulator += csr_values[i] * vector_values [csr_columns[i]];
    }

    output_values[index] = accumulator;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize, unsigned int nrows, const gpu::gpu_mem_32u& csr_row_offsets, const gpu::gpu_mem_32u& csr_columns, const gpu::gpu_mem_32u& csr_values, const gpu::gpu_mem_32u& vector_values, gpu::gpu_mem_32u& output_values) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        nrows, csr_row_offsets.cuptr(), csr_columns.cuptr(), csr_values.cuptr(), vector_values.cuptr(), output_values.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
