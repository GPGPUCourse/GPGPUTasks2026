#include "../defines.h"

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

// Один ворп суммирует одну строку, в рамках одной рабочей группы несколько ворпов = несколько строк
__global__ void sparse_csr_matrix_vector_multiplication(
    const uint* row_offsets,
    const uint* csr_columns,
    const uint* csr_values,
    const uint* vector,
    uint* result,
    const unsigned int nrows)
{
    const auto row = blockIdx.x * (blockDim.x / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    if (row >= nrows) {
        return;
    }
    const auto thread_offset = threadIdx.x % WARP_SIZE;

    const auto row_offset = row_offsets[row];
    const auto next_row_offset = row_offsets[row + 1];

    uint thread_acc = 0;
    for (auto i = row_offset + thread_offset; i < next_row_offset; i += WARP_SIZE) {
        const auto col = csr_columns[i];
        thread_acc += csr_values[i] * vector[col];
    }
    // __syncthreads здесь не нужен, так как мы в одном ворпе
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_acc += __shfl_down_sync(0xFFFFFFFF, thread_acc, offset);
    }

    if (thread_offset == 0) {
        result[row] = thread_acc;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_offsets, const gpu::gpu_mem_32u &csr_columns,
    const gpu::gpu_mem_32u &csr_values, const gpu::gpu_mem_32u &vector_values,
    gpu::gpu_mem_32u &result, unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(SMALL_GROUP_SIZE == WARP_SIZE, 17283561243, SMALL_GROUP_SIZE, WARP_SIZE);
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(), csr_columns.cuptr(), csr_values.cuptr(),
        vector_values.cuptr(), result.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
