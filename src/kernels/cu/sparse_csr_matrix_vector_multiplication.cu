#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Один варп обрабатывает одну строку матрицы
__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* csr_row_offsets,
    const unsigned int* csr_columns,
    const unsigned int* csr_values,
    const unsigned int* vector_values,
    unsigned int* output_vector_values,
    unsigned int nrows)
{
    const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = global_id / 32;
    const unsigned int lane = threadIdx.x % 32;

    // Все потоки лишнего варпа завершаются вместе
    if (row >= nrows) {
        return;
    }

    const unsigned int row_from = csr_row_offsets[row];
    const unsigned int row_to = csr_row_offsets[row + 1];

    // У каждого потока своя частичная сумма
    unsigned int sum = 0;
    for (unsigned int i = row_from + lane; i < row_to; i += 32) {
        const unsigned int col = csr_columns[i];
        sum += csr_values[i] * vector_values[col];
    }

    // Складываем частичные суммы всех 32 потоков варпа
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    // Только первый поток варпа записывает итог своей строки
    if (lane == 0) {
        output_vector_values[row] = sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& csr_row_offsets,
    const gpu::gpu_mem_32u& csr_columns,
    const gpu::gpu_mem_32u& csr_values,
    const gpu::gpu_mem_32u& vector_values,
    gpu::gpu_mem_32u& output_vector_values,
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