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
    // TODO
    if (i >= n)
        return;

    const int start = (i / (2 * sorted_k)) * (2 * sorted_k);
    const int middle = min(start + sorted_k, n);
    const int end = min(start + 2 * sorted_k, n);

    const bool is_left = i < middle;
    const unsigned int value = input_data[i];

    const int other_start = is_left ? middle : start;
    int low = other_start;
    int high = is_left ? end : middle;

    while (low < high) {
        const int mid = low + (high - low) / 2;

        if (input_data[mid] < value ||
            (!is_left && input_data[mid] == value)) {
            low = mid + 1;
            } else {
                high = mid;
            }
    }

    const int own_start = is_left ? start : middle;
    const int position = start + (i - own_start) + (low - other_start);

    output_data[position] = value;
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
