#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void sparse_csr_matrix_vector_multiplication_csr(
    const uint* row_offsets,
    const uint* cols,
    const uint* values,
    const uint n,
    const uint nrows,
    const uint* vtr,
    uint* dst
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // warp-per-row
    uint local_sum = 0;

    // Тогда нужно найти какая у нас строка
    const uint row = idx / 32;
    if (row >= nrows) {
        return;
    }
    const uint thread_idx = idx % 32;

    // Потом индекс ее начала и конца
    const uint offsets_from = row_offsets[row];
    const uint offsets_to   = row_offsets[row + 1];

    for (uint i = offsets_from; i < offsets_to; i += 128) {
        const uint idx_0 = i + thread_idx;
        const uint idx_1 = idx_0 + 32;
        const uint idx_2 = idx_0 + 64;
        const uint idx_3 = idx_0 + 96;

        uint col_0 = 0, valueM_0 = 0, valueV_0 = 0;
        if (idx_0 < offsets_to) {
            col_0 = cols[idx_0];
            valueM_0 = values[idx_0];
        }
        uint col_1 = 0, valueM_1 = 0, valueV_1 = 0;
        if (idx_1 < offsets_to) {
            col_1 = cols[idx_1];
            valueM_1 = values[idx_1];
        }
        uint col_2 = 0, valueM_2 = 0, valueV_2 = 0;
        if (idx_2 < offsets_to) {
            col_2 = cols[idx_2];
            valueM_2 = values[idx_2];
        }
        uint col_3 = 0, valueM_3 = 0, valueV_3 = 0;
        if (idx_3 < offsets_to) {
            col_3 = cols[idx_3];
            valueM_3 = values[idx_3];
        }
        /// Типа догрузились значения col, сразу отправляем пачку valV
        if (idx_0 < offsets_to) {
            valueV_0 = vtr[col_0];
        }
        if (idx_1 < offsets_to) {
            valueV_1 = vtr[col_1];
        }
        if (idx_2 < offsets_to) {
            valueV_2 = vtr[col_2];
        }
        if (idx_3 < offsets_to) {
            valueV_3 = vtr[col_3];
        }

        /// Жжем
        local_sum += valueM_0 * valueV_0;
        local_sum += valueM_1 * valueV_1;
        local_sum += valueM_2 * valueV_2;
        local_sum += valueM_3 * valueV_3;
    }

    // Теперь нужно просуммировать в варпе
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Но мы не будем
    if (thread_idx == 0) {
        dst[row] = local_sum;
    }
}

namespace cuda {
void sparse_csr_matrix_vector_multiplication_csr(
    gpu::gpu_mem_32u& row_offsets,
    gpu::gpu_mem_32u& cols,
    gpu::gpu_mem_32u& values,
    uint n,
    uint nrows,
    gpu::gpu_mem_32u& vtr,
    gpu::gpu_mem_32u& dst
) {
    gpu::Context context;
    WorkSize workSize(256, 32 * nrows);
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sparse_csr_matrix_vector_multiplication_csr<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        row_offsets.cuptr(),
        cols.cuptr(),
        values.cuptr(),
        n,
        nrows,
        vtr.cuptr(),
        dst.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
