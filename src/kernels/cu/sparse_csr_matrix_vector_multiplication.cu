#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void sparse_csr_matrix_vector_multiplication(
    const unsigned int* row_counters,
    const unsigned int* input_index,
    const unsigned int* input_value,
    const unsigned int* vector,
    unsigned int* output,
    unsigned int rows)
{
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const unsigned int begin = row_counters[row];
    const unsigned int end = row_counters[row + 1];

    unsigned int sum = 0;
    for (unsigned int index = begin; index < end; ++index) {
        sum += input_value[index] * vector[input_index[index]];
    }

    output[row] = sum;
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& row_counters,
    const gpu::gpu_mem_32u& input_index,
    const gpu::gpu_mem_32u& input_value,
    const gpu::gpu_mem_32u& vector,
    gpu::gpu_mem_32u& output,
    unsigned int rows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_counters.cuptr(),
        input_index.cuptr(),
        input_value.cuptr(),
        vector.cuptr(),
        output.cuptr(),
        rows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda