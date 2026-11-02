#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void merge_sort(
    const unsigned int* input_data,
    unsigned int* output_data,
    int sorted_k,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int pair_size = 2 * sorted_k;
    int begin = (i / pair_size) * pair_size;
    int middle = min(begin + sorted_k, n);
    int end = min(begin + pair_size, n);

    bool from_left = i < middle;
    int own_begin = from_left ? begin : middle;
    int other_begin = from_left ? middle : begin;
    int low = other_begin;
    int high = from_left ? end : middle;
    unsigned int value = input_data[i];

    while (low < high) {
        int mid = low + (high - low) / 2;
        if (input_data[mid] < value || (!from_left && input_data[mid] == value))
            low = mid + 1;
        else
            high = mid;
    }

    int own_rank = i - own_begin;
    int other_rank = low - other_begin;
    output_data[begin + own_rank + other_rank] = value;
}

namespace cuda {
void merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
