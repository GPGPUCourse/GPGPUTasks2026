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
    int pair = blockIdx.x * blockDim.x + threadIdx.x;

    int pair_size = 2 * sorted_k;
    int pair_count = (n + pair_size - 1) / pair_size;

    if (pair >= pair_count)
        return;

    int begin = pair * pair_size;
    int middle = min(begin + sorted_k, n);
    int end = min(begin + pair_size, n);

    int a = begin;
    int b = middle;
    int out = begin;

    while (a < middle && b < end) {
        if (input_data[a] <= input_data[b])
            output_data[out++] = input_data[a++];
        else
            output_data[out++] = input_data[b++];
    }

    while (a < middle)
        output_data[out++] = input_data[a++];

    while (b < end)
        output_data[out++] = input_data[b++];
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
