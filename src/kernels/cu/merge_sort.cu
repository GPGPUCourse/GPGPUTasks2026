#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ int find_rank(const unsigned int* data, int low, int high,
                         unsigned int value, bool is_first_half)
{
    while (low < high){
        int midle = (low + high) / 2;
        if(data[midle] < value){
            low = midle + 1;
        }
        else if (data[midle] == value){
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
    return low;
}

__global__ void merge_sort_borders(
    const unsigned int* input_data,
                   int* boards,
                   int  sorted_k,
                   int  n)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = g * GROUP_SIZE;
    if(first >= n) return;
    const int last = min(first + GROUP_SIZE, n) - 1;
    const int block_size = 2 * sorted_k;
    const int block_start = (first - first % block_size);
    bool is_first_half = !(((first - first % sorted_k) / sorted_k) % 2);
    int low = block_start + sorted_k * is_first_half;
    int high = min(low + sorted_k, n);
    boards[2 * g] = find_rank(input_data, low, high, input_data[first], is_first_half);
    boards[2 * g + 1] = find_rank(input_data, low, high, input_data[last], is_first_half);
}

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
    const          int* boards,
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
    if(sorted_k >= GROUP_SIZE){
        low = boards[2 * blockIdx.x];
        high = boards[2 * blockIdx.x + 1];
    }
    const uint value = input_data[i];
    low = find_rank(input_data, low, high, value, is_first_half);
    output_data[i + low - block_start - sorted_k] = value;
}

namespace cuda {
void merge_sort_borders(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32i &boards, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_borders<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), boards.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}

void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, const gpu::gpu_mem_32i &boards, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), boards.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
