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
    const unsigned int* vector,
    const unsigned int nnz,
    const unsigned int nrows,
    unsigned int* output_vector
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int l = -1;
    int r = nrows;
    while (l < r - 1) {
        int m = (static_cast<long long>(r) + l) / 2;
        if (row_offset[m] <= i) {
            l = m;
        }else {
            r = m;
        }
    }
    const unsigned int row = l;
    const unsigned int val = vector[columns[i]] * values[i];
    atomicAdd(output_vector + row, val);
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_offset,
    const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values,
    const gpu::gpu_mem_32u &vector,
    const unsigned int nnz,
    const unsigned int nrows,
    const gpu::gpu_mem_32u &output_vector){
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offset.cuptr(), columns.cuptr(), values.cuptr(), vector.cuptr(), nnz, nrows, output_vector.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
