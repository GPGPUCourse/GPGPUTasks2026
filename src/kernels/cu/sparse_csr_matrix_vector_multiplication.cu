#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int *csr_row_offsets,
    const unsigned int *csr_columns,
    const unsigned int *csr_values,
    const unsigned int *input_vector,
          unsigned int *output_vector,
    const unsigned int  nrows,
    const unsigned int  ncols)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int acc = 0;
    if(x < nrows){
        unsigned int curr_elem = csr_row_offsets[x];
        unsigned int limit = csr_row_offsets[x+1];
        for(; curr_elem < limit; curr_elem++){
            unsigned int curr_column = csr_columns[curr_elem];
            unsigned int curr_matrix_value = csr_values[curr_elem];
            unsigned int curr_vector_value = input_vector[curr_column];
            acc += curr_matrix_value * curr_vector_value;
        }
        output_vector[x] = acc;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &csr_row_offsets,
     const gpu::gpu_mem_32u &csr_columns,
     const gpu::gpu_mem_32u &csr_values,
     const gpu::gpu_mem_32u &input_vector,
           gpu::gpu_mem_32u &output_vector,
         const unsigned int  nrows,
         const unsigned int  ncols)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(),
        csr_columns.cuptr(),
        csr_values.cuptr(),
        input_vector.cuptr(),
        output_vector.cuptr(),
        nrows,
        ncols
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
