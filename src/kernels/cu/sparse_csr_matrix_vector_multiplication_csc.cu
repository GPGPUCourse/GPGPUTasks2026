#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication_csc(
    const uint* rows,
    const uint* col_offsets,
    const uint* values,
    const uint n,
    const uint ncols,
    const uint* vtr,
    uint* dst
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // warp-per-column

    // Тогда нужно найти какая у нас колонка
    const uint col = idx / 32;
    if (col >= ncols) {
        return;
    }
    const uint thread_idx = idx % 32;
    const uint valueV = vtr[col];

    // Потом индекс ее начала и конца
    const uint offsets_from = col_offsets[col];
    const uint offsets_to   = col_offsets[col + 1];

    for (uint i = offsets_from; i < offsets_to; i += 128) {
        const uint idx_0 = i + thread_idx;
        const uint idx_1 = idx_0 + 32;
        const uint idx_2 = idx_0 + 64;
        const uint idx_3 = idx_0 + 96;

        uint row_0 = 0, valueM_0 = 0;
        if (idx_0 < offsets_to) {
            row_0 = rows[idx_0];
            valueM_0 = values[idx_0];
        }
        uint row_1 = 0, valueM_1 = 0;
        if (idx_1 < offsets_to) {
            row_1 = rows[idx_1];
            valueM_1 = values[idx_1];
        }
        uint row_2 = 0, valueM_2 = 0;
        if (idx_2 < offsets_to) {
            row_2 = rows[idx_2];
            valueM_2 = values[idx_2];
        }
        uint row_3 = 0, valueM_3 = 0;
        if (idx_3 < offsets_to) {
            row_3 = rows[idx_3];
            valueM_3 = values[idx_3];
        }

        /// Жжем
        if (idx_0 < offsets_to) atomicAdd(&dst[row_0], valueV * valueM_0);
        if (idx_1 < offsets_to) atomicAdd(&dst[row_1], valueV * valueM_1);
        if (idx_2 < offsets_to) atomicAdd(&dst[row_2], valueV * valueM_2);
        if (idx_3 < offsets_to) atomicAdd(&dst[row_3], valueV * valueM_3);
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication_csc(
    gpu::gpu_mem_32u& rows,
    gpu::gpu_mem_32u& col_offsets,
    gpu::gpu_mem_32u& values,
    uint n,
    uint ncols,
    gpu::gpu_mem_32u& vtr,
    gpu::gpu_mem_32u& dst
) {
    gpu::Context context;
    WorkSize workSize(128, 32 * ncols);
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication_csc<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            rows.cuptr(),
            col_offsets.cuptr(),
            values.cuptr(),
            n,
            ncols,
            vtr.cuptr(),
            dst.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
