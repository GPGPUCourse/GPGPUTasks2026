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
    const unsigned int width = sorted_k;
    const unsigned int count = n;
    const unsigned int items = min(8u, 2 * width);
    const unsigned int start = (blockIdx.x * blockDim.x + threadIdx.x) * items;
    if (start >= count) {
        return;
    }

    const unsigned int base = start / (2 * width) * (2 * width);
    const unsigned int left_size = min(width, count - base);
    const unsigned int right_size = min(width, count - base - left_size);
    const unsigned int diagonal = start - base;
    const unsigned int* left = input_data + base;
    const unsigned int* right = left + left_size;
    unsigned int low = diagonal > right_size ? diagonal - right_size : 0;
    unsigned int high = min(diagonal, left_size);
    while (low < high) {
        const unsigned int a = low + (high - low) / 2;
        const unsigned int b = diagonal - a;
        if (b > 0 && a < left_size && right[b - 1] >= left[a]) {
            low = a + 1;
        } else {
            high = a;
        }
    }

    unsigned int a = low;
    unsigned int b = diagonal - a;
    const unsigned int end = start + min(items, count - start);
    for (unsigned int i = start; i < end; ++i) {
        if (a < left_size && (b >= right_size || left[a] <= right[b])) {
            output_data[i] = left[a++];
        } else {
            output_data[i] = right[b++];
        }
    }
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
