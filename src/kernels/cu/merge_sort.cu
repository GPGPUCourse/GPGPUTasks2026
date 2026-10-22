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
    if (i >= n) {
        return;
    }
    const auto el = input_data[i];

    const auto part_size = 1 << sorted_k;
    const bool is_left = ((i >> sorted_k) & 1) == 0;

    const int current_part_start = i - i % part_size;
    const int other_part_start = is_left ? current_part_start + part_size : current_part_start - part_size;

    int l = other_part_start - 1, r = min(other_part_start + part_size, n);
    // printf("%d (%s) -> ищем в [%d, %d)\n", i, is_left ? "лев" : "прв", other_part_start, r);

    while (r - l > 1) {
        const auto m = (l + r) / 2;
        const auto other_el = input_data[m];
        if (is_left ? other_el < el : other_el <= el) {
            l = m;
        } else {
            r = m;
        }
    }

    const auto res_index = is_left ? i + max(r - other_part_start, 0) : i + r - other_part_start - part_size;
    // printf("%d пишет %d в %d\n", i, el, res_index);
    output_data[res_index] = el;
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
