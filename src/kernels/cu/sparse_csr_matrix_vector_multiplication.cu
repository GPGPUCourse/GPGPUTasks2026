#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "helpers/spmv_impl.cuh"

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &row_offsets, const gpu::gpu_mem_32u &columns,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &vector_values,
    gpu::gpu_mem_32u &output, unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    if (nrows == 0) {
        return;
    }
    spmv7::multiply<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(), columns.cuptr(), values.cuptr(), vector_values.cuptr(), output.cuptr(), nrows);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
