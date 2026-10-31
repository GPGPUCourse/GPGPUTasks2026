#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <cuda/atomic>


__global__ void sparse_csr_matrix_vector_multiplication(
    const uint* row_offsets,
    const uint* csr_columns,
    const uint* csr_values,
    const uint* vector,
    uint* result,
    const unsigned int n)
{
    const auto row = blockIdx.x;

    const auto row_offset = row_offsets[row];
    const auto row_size = row_offsets[row + 1] - row_offset;

    uint thread_acc = 0;
    for (int row_block_offset = 0; row_block_offset < row_size; row_block_offset += WARP_SIZE) {
        if (row_block_offset + threadIdx.x < row_size) {
            const auto i = row_offset + row_block_offset + threadIdx.x;
            const auto col = csr_columns[i];
            const auto summand = csr_values[i] * vector[col];
            thread_acc += summand;
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_acc += __shfl_down_sync(0xFFFFFFFF, thread_acc, offset);
    }
    if (threadIdx.x == 0) {
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
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(), csr_columns.cuptr(), csr_values.cuptr(),
        vector_values.cuptr(), result.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
