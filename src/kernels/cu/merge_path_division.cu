#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"
#include <libbase/point.h>

__device__ __forceinline__ uint2 get_start_point(unsigned int d, unsigned int first_array_size)
{
    unsigned int y_res = min(d, first_array_size);
    return { d - y_res, y_res };
}

__device__ __forceinline__ uint2 get_end_point(unsigned int d, unsigned int second_array_size)
{
    unsigned int res_x = min(d, second_array_size);
    return {
        res_x, d - res_x
    };
}

__device__ __forceinline__ uint2 mid(uint2 left, uint2 right, unsigned int d)
{
    unsigned int x = (left.x + right.x) / 2;
    return {
        x,
        d - x,
    };
}

__device__ bool operator<(const uint2& a, const uint2& b)
{
    return a.x < b.x;
}

__device__ uint2& operator+(uint2& a, unsigned int x)
{
    a.x += x;
    a.y -= x;
    return a;
}

// -x + y_offset =

// formula for diagonal is -x + y_offset
__global__ void merge_path_division(
    const unsigned int* input_data, // input data consists of 2 subarrays to merge with offset of merge_array_size
    uint2* merge_subtasks,
    unsigned int merge_array_size,
    unsigned int n)
{

    unsigned int diagonal_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int diag_count = n / MERGE_PATH_DIAGONAL_OFFSET + (n % MERGE_PATH_DIAGONAL_OFFSET != 0 ? 1 : 0);
    if (diagonal_index >= diag_count) {
        return;
    }

    unsigned int task_size = merge_array_size * 2;
    unsigned int diag_count_per_task = task_size / MERGE_PATH_DIAGONAL_OFFSET + (task_size % MERGE_PATH_DIAGONAL_OFFSET != 0 ? 1 : 0);
    unsigned int diagonal_index_local = diagonal_index % diag_count_per_task;
    unsigned int task_index = diagonal_index / diag_count_per_task;

    unsigned int first_array_start = task_index * task_size;
    unsigned int first_array_size = min(merge_array_size, n - first_array_start);
    unsigned int second_array_start = first_array_start + first_array_size;
    unsigned int second_array_size = min(merge_array_size, n - second_array_start);

    unsigned int d = diagonal_index_local * MERGE_PATH_DIAGONAL_OFFSET;
    auto start_point = get_start_point(d, first_array_size);
    auto end_point = get_end_point(d, second_array_size);

    while (start_point < end_point) {
        auto mid_point = mid(start_point, end_point, d);
        unsigned int a = input_data[first_array_start + mid_point.y - 1];
        unsigned int b = input_data[second_array_start + mid_point.x];
        if (a > b) {
            start_point = mid_point + 1;
        } else {
            end_point = mid_point;
        }
    }
    merge_subtasks[diagonal_index] = start_point;
}

namespace cuda {
void merge_path_division(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::shared_device_buffer_typed<uint2>& diagonals, unsigned int merge_array_size, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_path_division<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), diagonals.cuptr(), merge_array_size, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
