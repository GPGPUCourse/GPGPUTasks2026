#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

enum class BoundMode {
    LOWER_BOUND,
    UPPER_BOUND
};

template <BoundMode Mode>
__device__ int binary_search_bound(const uint* array, int size, uint target) {
    int left = 0;
    int right = size;

    while (left < right) {
        int mid = left + (right - left) / 2;

        bool go_right;
        if (Mode == BoundMode::LOWER_BOUND) {
            go_right = (array[mid] < target);
        } else {
            go_right = (array[mid] <= target);
        }

        if (go_right) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left;
}

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const int pair_start = (idx / (2 * sorted_k)) * 2 * sorted_k;
    const int end_a  = min(pair_start + sorted_k, n);
    const int end_b = min(pair_start + 2 * sorted_k, n);

    const bool is_a = (idx < end_a);
    const unsigned int target_val = input_data[idx];
    int position;

    if (is_a) {
        position = idx + binary_search_bound<BoundMode::LOWER_BOUND>(input_data + end_a, end_b - end_a, target_val);
    } else {
        position = pair_start + (idx - end_a) + binary_search_bound<BoundMode::UPPER_BOUND>(input_data + pair_start, end_a - pair_start, target_val);
    }

    output_data[position] = target_val;
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
