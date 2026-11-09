#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void __launch_bounds__(GROUP_SIZE) merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const int pair_start = (i / (2 * sorted_k)) * (2 * sorted_k);
    const bool is_left = (i / sorted_k) % 2 == 0;
    const int own_start = is_left ? pair_start : pair_start + sorted_k;
    const int other_start = is_left ? pair_start + sorted_k : pair_start;
    const int other_end = min(other_start + sorted_k, n);

    const unsigned int value = input_data[i];
    int lo = other_start;
    int hi = other_end;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (input_data[mid] < value || (!is_left && input_data[mid] == value)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int rank = lo - other_start;

    output_data[pair_start + (i - own_start) + rank] = value;
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
