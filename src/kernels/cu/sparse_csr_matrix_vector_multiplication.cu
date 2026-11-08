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
    unsigned int* result_vector_values,
    unsigned int nrows, unsigned int ncols)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int accumulator = 0;
    unsigned int row_from = csr_row_offsets[row];
    unsigned int row_to = csr_row_offsets[row + 1];
    for (unsigned int i = row_from; i < row_to; ++i) {
        unsigned int col = csr_columns[i];
        accumulator += csr_values[i] * vector_values[col];
    }
    result_vector_values[row] = accumulator;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &csr_row_offsets, gpu::gpu_mem_32u &csr_columns, 
    gpu::gpu_mem_32u &csr_values, gpu::gpu_mem_32u &vector_values,
    gpu::gpu_mem_32u &result_vector_values,
    unsigned int nrows, unsigned int ncols
) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(), csr_columns.cuptr(), 
        csr_values.cuptr(), vector_values.cuptr(), 
        result_vector_values.cuptr(), 
        nrows, ncols
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
