#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ unsigned int lower_bound(const unsigned int* arr, unsigned int l, unsigned int r, unsigned int value) {
    while (l < r) {
        unsigned int mid = l + (r - l) / 2;
        if (value > arr[mid]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__device__ unsigned int upper_bound(const unsigned int* arr, unsigned int l, unsigned int r, unsigned int value) {
    while (l < r) {
        unsigned int mid = l + (r - l) / 2;
        if (value >= arr[mid]) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
          unsigned int  n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    unsigned int pair_start = (index / (2 * sorted_k)) * 2 * sorted_k;
    unsigned int mid = min(pair_start + sorted_k, n);//<-
    unsigned int end = min(pair_start + 2 * sorted_k, n);//<-
    unsigned int pair_id;
    unsigned int local_id;
    unsigned int value = input_data[index];

    if (index < mid) {
        pair_id = lower_bound(input_data, mid, end, value) - mid;
        local_id = index - pair_start;
    } else {
        pair_id = upper_bound(input_data, pair_start, mid, value) - pair_start;
        local_id = index - mid;
    }
    output_data[pair_start + pair_id + local_id] = value;
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
