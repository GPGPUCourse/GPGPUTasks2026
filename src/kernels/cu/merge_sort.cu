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

    unsigned int block = i / sorted_k;
    bool even = block % 2 == 0;

    unsigned int l = block;
    if (even)
        ++l;
    else
        --l;
    unsigned int r = l + 1;
    l *= sorted_k, r *= sorted_k;
    r = min(r, n);
    unsigned int block1_start = block * sorted_k;
    unsigned int block2_start = l;

    if (max(block1_start, block2_start) >= n) {
        output_data[i] = input_data[i];
        return;
    }

    unsigned int val = input_data[i];
    while (l < r) {
        unsigned int m = (l + r) / 2;
        unsigned int x = input_data[m];
        if ((x < val) || (!even && x == val))
            l = m + 1;
        else
            r = m;
    }
    unsigned int index = min(block1_start, block2_start) + (i - block1_start) + (l - block2_start);

    output_data[index] = val;
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
