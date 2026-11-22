#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Each block has SMALL_GROUP_SIZE threads (32 threads, one warp).
__global__ void sparse_csr_matrix_vector_multiplication(
    unsigned* csr_row_offsets,
    unsigned* csr_columns,
    unsigned* csr_values,
    unsigned* vector_values,
    unsigned* output_vector_values_gpu)
{
    unsigned out_vec_i = blockIdx.x;
    unsigned loc_id = threadIdx.x;
    unsigned row_start = csr_row_offsets[out_vec_i];
    unsigned row_end = csr_row_offsets[out_vec_i + 1];

    unsigned loc_acc = 0;
    for (unsigned chunk_start = row_start; chunk_start < row_end; chunk_start += SMALL_GROUP_SIZE) {
        unsigned idx = chunk_start + loc_id;
        if (idx < row_end)
            loc_acc += csr_values[idx] * vector_values[csr_columns[idx]];
    }

    for (unsigned stride = SMALL_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        loc_acc += __shfl_down_sync(0xffffffffu, loc_acc, stride);
    }

    if (loc_id == 0) {
        output_vector_values_gpu[out_vec_i] = loc_acc;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize& workSize,
    gpu::gpu_mem_32u& csr_row_offsets_gpu, gpu::gpu_mem_32u& csr_columns_gpu, gpu::gpu_mem_32u& csr_values_gpu,
    gpu::gpu_mem_32u& vector_values_gpu, gpu::gpu_mem_32u& output_vector_values_gpu)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const dim3 block = workSize.cuBlockSize();
    rassert(block.x == 32 && block.y == 1 && block.z == 1,
        4830257631, block.x, block.y, block.z);
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets_gpu.cuptr(), csr_columns_gpu.cuptr(), csr_values_gpu.cuptr(), vector_values_gpu.cuptr(), output_vector_values_gpu.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
