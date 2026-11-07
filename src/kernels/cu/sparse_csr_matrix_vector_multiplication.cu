#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int *row_offsets,
    const unsigned int *columns,
    const unsigned int *values,
    const unsigned int *vector_values,
    unsigned int *output, unsigned int nrows)
{
    extern __shared__ unsigned int partial_sums[];
    const unsigned int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int sum = 0;

    if (row < nrows) {
        const unsigned int end = row_offsets[row + 1];
        for (size_t i = size_t(row_offsets[row]) + lane; i < end; i += WARP_SIZE) {
            sum += values[i] * vector_values[columns[i]];
        }
    }
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        if (lane < offset)
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];

        __syncthreads();
    }

    if (row < nrows && lane == 0)
        output[row] = partial_sums[threadIdx.x];
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_offsets, const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &vector_values,
    gpu::gpu_mem_32u &output, unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(workSize.cuBlockSize().x % WARP_SIZE == 0 &&
            workSize.cuBlockSize().y == 1 && workSize.cuBlockSize().z == 1 &&
            workSize.cuGridSize().y == 1 && workSize.cuGridSize().z == 1, 78123650);
    if (nrows == 0)
        return;
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), workSize.cuBlockSize().x * sizeof(unsigned int), stream>>>(row_offsets.cuptr(), columns.cuptr(), values.cuptr(), vector_values.cuptr(), output.cuptr(), nrows);
    CUDA_CHECK_KERNEL_SYNC(stream);
}
} // namespace cuda
