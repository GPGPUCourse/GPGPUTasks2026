#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* csr_row_offsets, // размер nrows + 1, префиксные суммы числа ненулевых элементов по строкам
    const unsigned int* csr_cols, // размер nnz, номера столбцов ненулевых элементов
    const unsigned int* csr_vals, // размер nnz, значения ненулевых элементов
    const unsigned int* vector_values, // размер ncols, умножаемый вектор
    unsigned int* out, // размер nrows, результат
    unsigned int nrows,
    unsigned int ncols)
{
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= nrows)
        return;
    const auto rowFrom = csr_row_offsets[row];
    const auto rowTo = csr_row_offsets[row + 1];
    unsigned int acc = 0;
    for (size_t i = rowFrom; i < rowTo; ++i) {
        unsigned int col = csr_cols[i];
        acc += csr_vals[i] * vector_values[col];
    }

    out[row] = acc;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& csr_row_offsets,
    const gpu::gpu_mem_32u& csr_cols,
    const gpu::gpu_mem_32u& csr_vals,
    const gpu::gpu_mem_32u& vector_values,
    gpu::gpu_mem_32u& output_vector_values,
    unsigned int nrows,
    unsigned int ncols)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets.cuptr(),
        csr_cols.cuptr(),
        csr_vals.cuptr(),
        vector_values.cuptr(),
        output_vector_values.cuptr(),
        nrows,
        ncols);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
