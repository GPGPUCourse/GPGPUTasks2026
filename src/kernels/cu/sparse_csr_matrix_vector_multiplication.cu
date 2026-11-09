#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../kernels.h"

namespace {

using uint = unsigned int;

template<bool STREAMING>
__device__ __forceinline__ uint load_matrix(const uint* address)
{
    return STREAMING ? __ldcs(address) : __ldg(address);
}

template<int THREADS_PER_ROW, bool STREAMING, int FIXED_NNZ>
__global__ void spmv_optimized(
    const uint* __restrict__ row_offsets,
    const uint* __restrict__ columns,
    const uint* __restrict__ values,
    const uint* __restrict__ x,
    uint* __restrict__ y,
    uint nrows)
{
    const uint row = blockIdx.x * (blockDim.x / THREADS_PER_ROW)
        + threadIdx.x / THREADS_PER_ROW;
    const uint lane = threadIdx.x & (THREADS_PER_ROW - 1);
    const uint group_start = (threadIdx.x & 31u) - lane;
    const uint mask = (0xffffffffu >> (32 - THREADS_PER_ROW)) << group_start;

    if (row >= nrows) {
        return;
    }

    uint sums[4] = {0, 0, 0, 0};

    // FIXED_NNZ используется только после проверки всех длин строк на CPU.
    if (FIXED_NNZ != 0) {
        const uint begin = row * FIXED_NNZ;
        #pragma unroll
        for (int k = 0; k < FIXED_NNZ / THREADS_PER_ROW; ++k) {
            const uint i = begin + lane + k * THREADS_PER_ROW;
            const uint col = load_matrix<STREAMING>(columns + i);
            const uint value = load_matrix<STREAMING>(values + i);
            sums[k & 3] += value * __ldg(x + col);
        }
    } else {
        const uint begin = load_matrix<STREAMING>(row_offsets + row);
        const uint length = load_matrix<STREAMING>(row_offsets + row + 1) - begin;

        for (uint base = 0; base < length; base += 4 * THREADS_PER_ROW) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint relative = base + lane + k * THREADS_PER_ROW;
                if (relative < length) {
                    const uint i = begin + relative;
                    const uint col = load_matrix<STREAMING>(columns + i);
                    const uint value = load_matrix<STREAMING>(values + i);
                    sums[k] += value * __ldg(x + col);
                }
            }
        }
    }

    uint sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset, THREADS_PER_ROW);
    }

    if (lane == 0) {
        __stcs(y + row, sum);
    }
}

__global__ void spmv_baseline(
    const uint* row_offsets,
    const uint* columns,
    const uint* values,
    const uint* x,
    uint* y,
    uint nrows)
{
    const uint global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = global_id / 32;
    const uint lane = threadIdx.x % 32;

    if (row >= nrows) {
        return;
    }

    const uint begin = row_offsets[row];
    const uint end = row_offsets[row + 1];
    uint sum = 0;
    for (uint i = begin + lane; i < end; i += 32) {
        sum += values[i] * x[columns[i]];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

struct SpmvBuffers {
    const uint* row_offsets;
    const uint* columns;
    const uint* values;
    const uint* x;
    uint* y;
};

template<int THREADS_PER_ROW, bool STREAMING>
void launch_optimized(
    const cuda::SpmvConfig& config, const SpmvBuffers& a,
    uint nrows, cudaStream_t stream)
{
    const uint rows_per_block = config.block_size / THREADS_PER_ROW;
    const uint blocks = (nrows - 1) / rows_per_block + 1;

    if (config.fixed_nnz_per_row == 32) {
        spmv_optimized<THREADS_PER_ROW, STREAMING, 32>
            <<<blocks, config.block_size, 0, stream>>>(
                a.row_offsets, a.columns, a.values, a.x, a.y, nrows);
    } else if (config.fixed_nnz_per_row == 128) {
        spmv_optimized<THREADS_PER_ROW, STREAMING, 128>
            <<<blocks, config.block_size, 0, stream>>>(
                a.row_offsets, a.columns, a.values, a.x, a.y, nrows);
    } else {
        spmv_optimized<THREADS_PER_ROW, STREAMING, 0>
            <<<blocks, config.block_size, 0, stream>>>(
                a.row_offsets, a.columns, a.values, a.x, a.y, nrows);
    }
}

template<bool STREAMING>
void launch_by_width(
    const cuda::SpmvConfig& config, const SpmvBuffers& a,
    uint nrows, cudaStream_t stream)
{
    switch (config.threads_per_row) {
        case 8:
            launch_optimized<8, STREAMING>(config, a, nrows, stream);
            break;
        case 16:
            launch_optimized<16, STREAMING>(config, a, nrows, stream);
            break;
        case 32:
            launch_optimized<32, STREAMING>(config, a, nrows, stream);
            break;
    }
}

} // namespace

namespace cuda {

void sparse_csr_matrix_vector_multiplication(
    const SpmvConfig& config,
    const gpu::gpu_mem_32u& csr_row_offsets,
    const gpu::gpu_mem_32u& csr_columns,
    const gpu::gpu_mem_32u& csr_values,
    const gpu::gpu_mem_32u& vector_values,
    gpu::gpu_mem_32u& output_vector_values,
    unsigned int nrows)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(config.threads_per_row == 8 || config.threads_per_row == 16
        || config.threads_per_row == 32, 724001, config.threads_per_row);
    rassert(config.block_size == 128 || config.block_size == 256
        || config.block_size == 512, 724002, config.block_size);
    rassert(config.fixed_nnz_per_row == 0 || config.fixed_nnz_per_row == 32
        || config.fixed_nnz_per_row == 128, 724003, config.fixed_nnz_per_row);

    if (nrows == 0) {
        return;
    }

    const cudaStream_t stream = context.cudaStream();
    const SpmvBuffers a = {
        csr_row_offsets.cuptr(), csr_columns.cuptr(), csr_values.cuptr(),
        vector_values.cuptr(), output_vector_values.cuptr()
    };

    if (config.baseline) {
        const uint rows_per_block = config.block_size / 32;
        const uint blocks = (nrows - 1) / rows_per_block + 1;
        spmv_baseline<<<blocks, config.block_size, 0, stream>>>(
            a.row_offsets, a.columns, a.values, a.x, a.y, nrows);
    } else if (config.streaming_loads) {
        launch_by_width<true>(config, a, nrows, stream);
    } else {
        launch_by_width<false>(config, a, nrows, stream);
    }

    CUDA_CHECK_KERNEL_SYNC(stream);
}

} // namespace cuda
