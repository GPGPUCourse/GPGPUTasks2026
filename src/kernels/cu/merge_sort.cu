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
    if(i >= n) return;
    const int block_size = 2 * sorted_k;
    const int block_start = (i - i % block_size);
    bool is_first_half = !(((i - i % sorted_k) / sorted_k) % 2);
    int low = block_start + sorted_k * is_first_half;
    int high = min(low + sorted_k, n);
    const uint value = input_data[i];
    while (low < high){
        int midle = (low + high) / 2;
        if(input_data[midle] < value){
            low = midle + 1;
        }
        else if (input_data[midle] == value){
            if(is_first_half){
                high = midle;
            }
            else{
                low = midle + 1;
            }
        }
        else{
            high = midle;
        }
    }
    output_data[i + low - block_start - sorted_k] = value;
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
