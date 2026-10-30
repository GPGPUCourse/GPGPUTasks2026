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

    if (blockIdx.x * 4 >= diag_count) {
        return;
    }

    unsigned int group_index = threadIdx.x / 8;
    unsigned int thread_index = threadIdx.x % 8;

    unsigned int diag_index = blockIdx.x * 4 + group_index;
    unsigned int diag_index_local = diag_index % diag_count_local;
    unsigned int task_index = diag_index / diag_count_local;

    bool active = diag_index < diag_count;

    unsigned int first_array_start = task_index * task_size;
    unsigned int first_array_size = 0;
    unsigned int second_array_start = 0;
    unsigned int second_array_size = 0;

    if (active) {
        first_array_size = min(merge_array_size, n - first_array_start);
        second_array_start = first_array_start + first_array_size;
        second_array_size = min(merge_array_size, n - second_array_start);
    }

    __shared__ uint2 start[4];
    __shared__ uint2 end[4];

    __shared__ unsigned int first[4][MERGE_PATH_DIAGONAL_OFFSET];
    __shared__ unsigned int second[4][MERGE_PATH_DIAGONAL_OFFSET];
    __shared__ unsigned int res[4][MERGE_PATH_DIAGONAL_OFFSET];

    if (thread_index == 0 && active) {
        start[group_index] = merge_subtasks[diag_index];
        end[group_index] = { second_array_size, first_array_size };

        if (diag_index_local + 1 < diag_count_local && diag_index + 1 < diag_count) {
            end[group_index] = merge_subtasks[diag_index + 1];
        }
    }

    unsigned int first_size = 0;
    unsigned int second_size = 0;

    if (active) {
        first_size = end[group_index].y - start[group_index].y;
        second_size = end[group_index].x - start[group_index].x;
    }

    for (unsigned int i = thread_index; i < first_size; i += 8) {
        first[group_index][i] = input_data[first_array_start + start[group_index].y + i];
    }

    for (unsigned int i = thread_index; i < second_size; i += 8) {
        second[group_index][i] = input_data[second_array_start + start[group_index].x + i];
    }

    if (thread_index == 0) {
        unsigned int l = 0, r = 0;

        for (unsigned int k = 0; k < first_size + second_size; ++k) {
            if (l < first_size && (r == second_size || first[group_index][l] <= second[group_index][r])) {
                res[group_index][k] = first[group_index][l++];
            } else {
                res[group_index][k] = second[group_index][r++];
            }
        }
    }

    for (unsigned int i = thread_index; i < first_size + second_size; i += 8) {
        output_data[first_array_start + diag_index_local * MERGE_PATH_DIAGONAL_OFFSET + i] = res[group_index][i];
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
