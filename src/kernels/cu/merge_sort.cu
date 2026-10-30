#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void merge_sort(
    const unsigned int* input_data,
    uint2* merge_subtasks,
    unsigned int* output_data,
    unsigned int merge_array_size,
    unsigned int n)
{
    unsigned int task_size = merge_array_size * 2;

    unsigned int diag_count_local = task_size / MERGE_PATH_DIAGONAL_OFFSET + (task_size % MERGE_PATH_DIAGONAL_OFFSET != 0 ? 1 : 0);

    unsigned int diag_count = n / MERGE_PATH_DIAGONAL_OFFSET + (n % MERGE_PATH_DIAGONAL_OFFSET != 0 ? 1 : 0);
    unsigned int diag_index = blockIdx.x;

    if (diag_index >= diag_count) {
        return;
    }

    unsigned int diag_index_local = diag_index % diag_count_local;
    unsigned int task_index = blockIdx.x / diag_count_local;

    unsigned int first_array_start = task_index * task_size;
    unsigned int first_array_size = min(merge_array_size, n - first_array_start);
    unsigned int second_array_start = first_array_start + first_array_size;
    unsigned int second_array_size = min(merge_array_size, n - second_array_start);

    __shared__ uint2 start;
    __shared__ uint2 end;
    if (threadIdx.x == 0) {
        start = merge_subtasks[diag_index];
        end = { second_array_size, first_array_size };
        if (diag_index_local + 1 < diag_count_local && diag_index + 1 < diag_count) {
            end = merge_subtasks[diag_index + 1];
        }
    }
    __syncthreads();
    __shared__ unsigned int first[MERGE_PATH_DIAGONAL_OFFSET];
    __shared__ unsigned int second[MERGE_PATH_DIAGONAL_OFFSET];
    __shared__ unsigned int res[MERGE_PATH_DIAGONAL_OFFSET * 2];
    unsigned int first_size = end.y - start.y;
    unsigned int second_size = end.x - start.x;

    for (unsigned int i = threadIdx.x; i < first_size; i += blockDim.x) {
        first[i] = input_data[first_array_start + start.y + i];
    }
    for (unsigned int i = threadIdx.x; i < second_size; i += blockDim.x) {
        second[i] = input_data[second_array_start + start.x + i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int l = 0, r = 0;

        for (unsigned int k = 0; k < first_size + second_size; ++k) {
            if (l < first_size && (r == second_size || first[l] <= second[r])) {
                res[k] = first[l++];
            } else {
                res[k] = second[r++];
            }
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < first_size + second_size; i += blockDim.x) {
        output_data[first_array_start + diag_index_local * MERGE_PATH_DIAGONAL_OFFSET + i] = res[i];
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, gpu::shared_device_buffer_typed<uint2>& diagonals, unsigned int merge_array_size, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), diagonals.cuptr(), output_data.cuptr(), merge_array_size, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
