#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__inline__ __device__
uint32_t warp_reduce_sum(uint32_t val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void sparse_csr_matrix_vector_multiplication(const uint32_t *csr_row_offsets, const uint32_t *csr_columns, const uint32_t *csr_values, const uint32_t *vector_values, uint32_t *result_vector_values, int nrows, int ncols) // TODO input/output buffers
{
    uint32_t vec_i = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (vec_i >= nrows) return;
    
    uint32_t warp_tid = threadIdx.x % WARP_SIZE; 
    uint32_t mat_row = vec_i;
    uint32_t row_begin = csr_row_offsets[mat_row];
    uint32_t row_end = csr_row_offsets[mat_row+1];
    
    uint32_t result = 0;
    for (int col_offset = warp_tid; col_offset < row_end-row_begin; col_offset += WARP_SIZE) {
      uint32_t mat_col = csr_columns[row_begin+col_offset];
      uint32_t mat_val = csr_values[row_begin+col_offset];
      uint32_t vec_elem = vector_values[mat_col];
      result += mat_val * vec_elem;
    }
    result = warp_reduce_sum(result);

    if (warp_tid == 0) {
      result_vector_values[vec_i] = result;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &csr_row_offsets_gpu, const gpu::gpu_mem_32u &csr_columns_gpu, const gpu::gpu_mem_32u &csr_values_gpu, const gpu::gpu_mem_32u &vector_values_gpu, gpu::gpu_mem_32u &result_vector_values_gpu, int nrows, int ncols) // TODO input/output buffers
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        csr_row_offsets_gpu.cuptr(), csr_columns_gpu.cuptr(), csr_values_gpu.cuptr(), vector_values_gpu.cuptr(), result_vector_values_gpu.cuptr(), nrows, ncols
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
