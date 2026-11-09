#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* row_offset,
    const unsigned int* columns,
    const unsigned int* values,
    const unsigned int* vector_values,
    unsigned int* output_vector,
    unsigned int nrows
)
{
    const unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= nrows) {
        return;
    }

    unsigned int sum = 0;
    for (unsigned int i = row_offset[row]; i < row_offset[row+1]; i++) {
        sum += values[i] * vector_values[columns[i]];
    }
    output_vector[row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize &workSize,     
    const gpu::gpu_mem_32u &row_offset,
    const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &vector_values,
    gpu::gpu_mem_32u &output_vector,
    unsigned int nrows
)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offset.cuptr(), columns.cuptr(), values.cuptr(), vector_values.cuptr(), output_vector.cuptr(), nrows
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
