#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const uint32_t* __restrict__ row_offsets,
    const uint32_t* __restrict__ columns,
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ vector,
    uint32_t* __restrict__ result,
    size_t rows)
{
    __shared__ uint32_t results[GROUP_SIZE];

    const size_t tid = threadIdx.x;
    const size_t off = tid % SMALL_GROUP_SIZE;
    const size_t row = (blockIdx.x * blockDim.x + tid) / SMALL_GROUP_SIZE;

    uint32_t sum = 0;
    if (row < rows) {
        const size_t start = row_offsets[row];
        const size_t end   = row_offsets[row + 1];
        for (size_t i = start + off; i < end; i += SMALL_GROUP_SIZE) {
            sum += values[i] * vector[columns[i]];
        }
    }

    results[tid] = sum;
    __syncthreads();

    if (off == 0 && row < rows) {
#pragma unroll
        for (size_t i = 1; i < SMALL_GROUP_SIZE; ++i) {
            sum += results[tid + i];
        }
        result[row] = sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u& csr_row_offsets_gpu, const gpu::gpu_mem_32u& csr_columns_gpu, const gpu::gpu_mem_32u& csr_values_gpu, const gpu::gpu_mem_32u& vector_values_gpu, gpu::gpu_mem_32u& output_vector_values_gpu, uint32_t rows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets_gpu.cuptr(),
        csr_columns_gpu.cuptr(),
        csr_values_gpu.cuptr(),
        vector_values_gpu.cuptr(),
        output_vector_values_gpu.cuptr(),
        rows
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
