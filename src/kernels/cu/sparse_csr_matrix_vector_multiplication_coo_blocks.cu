#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication_coo_blocks(
    const uint* rows,
    const uint* cols,
    const uint* values,
    const uint* bucket_offsets,
    const uint ncols,
    const uint* vtr,
    uint* dst
) {
    __shared__ uint vtr_small[STRIP_WIDTH];

    // Подгружаем в локальную память
    const uint dst_index = threadIdx.x;
    const uint src_index = blockIdx.x * blockDim.x + dst_index;
    if (dst_index < STRIP_WIDTH) {
        vtr_small[dst_index] = vtr[src_index];
    } else {
        vtr_small[dst_index] = 0;
    }

    __syncthreads();

    const uint offset_from = bucket_offsets[blockIdx.x];
    const uint offset_to   = bucket_offsets[blockIdx.x + 1];

    for (uint i = offset_from; i < offset_to; i += blockDim.x) {
        uint idx = i + threadIdx.x;
        if (idx < offset_to) {
            atomicAdd(&dst[rows[idx]], values[idx] * vtr_small[cols[idx] % STRIP_WIDTH]);
        }
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication_coo_blocks(
    gpu::gpu_mem_32u& rows,
    gpu::gpu_mem_32u& cols,
    gpu::gpu_mem_32u& values,
    gpu::gpu_mem_32u& bucket_offsets,
    uint ncols,
    gpu::gpu_mem_32u& vtr,
    gpu::gpu_mem_32u& dst
) {
    gpu::Context context;
    const uint n_strips = div_ceil(ncols, (uint) STRIP_WIDTH);
    WorkSize workSize(STRIP_WIDTH, n_strips * STRIP_WIDTH);
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication_coo_blocks<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            rows.cuptr(),
            cols.cuptr(),
            values.cuptr(),
            bucket_offsets.cuptr(),
            ncols,
            vtr.cuptr(),
            dst.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
