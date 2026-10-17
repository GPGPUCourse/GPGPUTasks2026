#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define V SCATTER_THREAD_WORK

#include "helpers/uint_vectors.cu"

__global__ void radix_sort_03_scatter(
    const uintV* source,
    const uint* counts,
          uint* result,
    const uint offset,
    const uint zero_count,
    const uint n)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;

    const uintV source_value = x < n ? source[x] : uintV{};
    const uintV is_zero = x < n ? ((source_value >> offset) & 1) ^ 1 : uintV{};
    uintV is_zero_sums = is_zero;
    calc_vector_prefsums(is_zero_sums);

    // Считаем исключительные локальные префиксные суммы кол-ва нулей
    __shared__ uint local_counts[SCATTER_GROUP_SIZE];
    local_counts[threadIdx.x] = is_zero_sums.array[V - 1];
    __syncthreads();
    if (threadIdx.x == 0) {
        uint sum = 0;
        for (int i = 0; i < SCATTER_GROUP_SIZE; ++i) {
            const auto val = local_counts[i];
            local_counts[i] = sum;
            sum += val;
        }
    }
    __syncthreads();

    if (x >= n) {
        return;
    }

    const auto local_zeroes_before = (blockIdx.x == 0 ? 0 : counts[blockIdx.x - 1]) + local_counts[threadIdx.x];
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const auto zeroes_before = local_zeroes_before + is_zero_sums.array[i];
        if (is_zero.array[i] == 1) {
            result[zeroes_before - 1] = source_value.array[i];
        } else {
            const auto ones_before = x * V + i - zeroes_before;
            result[zero_count + ones_before] = source_value.array[i];
        }
    }
}

namespace cuda {
void radix_sort_03_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& source, const gpu::gpu_mem_32u& counts, gpu::gpu_mem_32u& result, uint n, uint offset, uint total_zero_count)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    rassert(n % V == 0, 186523419837, n, V);
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        reinterpret_cast<const uintV*>(source.cuptr()),
        counts.cuptr(),
        result.cuptr(),
        offset,
        total_zero_count,
        n / V);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
