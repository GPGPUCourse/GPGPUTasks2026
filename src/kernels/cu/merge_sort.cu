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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int my_block = i / sorted_k;
    unsigned int rival_block = my_block ^ 1;
    unsigned int tiebreaker = my_block & 1;
    unsigned int me = input_data[i];

    unsigned int l = rival_block * sorted_k;
    unsigned int r = rival_block * sorted_k + sorted_k;
    r = min(r, n);
    // output_data[i] = l;
    // return;
    if (l >= n) {
        output_data[i] = me;
        return;
    }
    if ((tiebreaker == 0 && me <= input_data[l]) ||
        (tiebreaker == 1 && me < input_data[l])) {
        // output_data[i] = i - sorted_k * (my_block & 1);
        output_data[i - sorted_k * (my_block & 1)] = me;
        return;
    }

    while (r - l > 1) {
        unsigned int mid = (l + r) / 2;
        unsigned int rival = input_data[mid];
        if ((tiebreaker == 0 && me <= rival) ||
            (tiebreaker == 1 && me < rival)) {
            r = mid;
        } else {
            l = mid;
        }
    }


    unsigned int final_position = i + (r - rival_block * sorted_k) - sorted_k * (my_block & 1);
    // output_data[i] = final_position;
    output_data[final_position] = me;
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
