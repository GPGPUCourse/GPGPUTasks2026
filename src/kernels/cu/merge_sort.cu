#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const int left_begin = (i / (2 * sorted_k)) * (2 * sorted_k);
    const int right_begin = min(left_begin + sorted_k, n);
    const int right_end = min(left_begin + 2 * sorted_k, n);

    const unsigned int x = input_data[i];
    const bool in_left = (i < right_begin);

    int l = left_begin;
    int r = right_begin;
    if (in_left)
    {
        l = right_begin;
        r = right_end;
    }
    const int other_begin = l;
    while (l < r) {
        const int mid = (l + r) / 2;
        const unsigned int y = input_data[mid];
        if (y < x || (!in_left && y == x)) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    const int count_before_in_other = l - other_begin;
    const int index_in_own = in_left ? (i - left_begin) : (i - right_begin);

    output_data[left_begin + index_in_own + count_before_in_other] = x;
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
