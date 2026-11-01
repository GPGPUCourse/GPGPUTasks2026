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
    // global
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    const int start_index = index - index % sorted_k;
    
    // in block
    const bool is_left = index % sorted_k < (sorted_k / 2);
    const int other_start_index = is_left ? start_index + (sorted_k / 2) : start_index;

    const int in_index = index % (sorted_k / 2);

    // find
    uint num = input_data[index];

    uint l = 0;
    uint r = sorted_k / 2;

    if(input_data[other_start_index] > num || (is_left && input_data[other_start_index] == num)) {
        r = 0;
    }

    while(r - l > 1) {
        uint m = (l + r) / 2;
        if(input_data[other_start_index + m] > num) {
            r = m;
        } else if(input_data[other_start_index + m] < num) {
            l = m;
        } else {
            if(is_left) {
                r = m;
            } else {
                l = m;
            }
        }
    }

    // out
    output_data[start_index + in_index + r] = input_data[index];
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
