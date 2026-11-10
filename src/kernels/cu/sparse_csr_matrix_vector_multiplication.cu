#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication
    (const uint *csr_row_offsets, const uint *csr_columns, const uint *csr_values,
    const uint* vector_values, uint *output, uint vector_size, uint n)
{
    const uint workitem_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (workitem_index < vector_size) {
        const uint start_index = csr_row_offsets[workitem_index];
        const uint end_index = csr_row_offsets[workitem_index + 1];
        uint res = 0;
        for (int i = start_index; i < end_index; i++) {
            res += csr_values[i] * vector_values[csr_columns[i]];
        }
        output[workitem_index] = res;
    }
    return;

}

namespace cuda {
void sparse_csr_matrix_vector_multiplication
    (const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &csr_row_offsets_gpu,
    const gpu::gpu_mem_32u &csr_columns, const gpu::gpu_mem_32u &csr_values,
    const gpu::gpu_mem_32u &vector_values, gpu::gpu_mem_32u &output, uint vector_size, uint n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets_gpu.cuptr(), csr_columns.cuptr(), csr_values.cuptr(), vector_values.cuptr(), output.cuptr(), vector_size, n
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
