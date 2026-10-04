#include <cmath>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* input,
    unsigned int* output,
    unsigned int  n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    if (index < n) {
        local_data[local_index] = input[index];
    } else {
        local_data[local_index] = 0;
    }
    __syncthreads();
    
    uint block_sum = 0;
    if (local_index == 0) {
        
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            block_sum += local_data[i];
        }
        
        output[blockIdx.x] = block_sum;

    }
    

    // TODO
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, gpu::gpu_mem_32u &reduction_buffer1_gpu, gpu::gpu_mem_32u &reduction_buffer2_gpu, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    unsigned int current_n = n;
    gpu::gpu_mem_32u current_input = a;
    gpu::gpu_mem_32u current_output = reduction_buffer1_gpu;
    gpu::WorkSize current_workSize = workSize;
    int is_even_iteration = 1;
    while (current_n > 1) {
        ::sum_04_local_reduction<<<current_workSize.cuGridSize(), current_workSize.cuBlockSize(), 0, stream>>>(current_input.cuptr(), current_output.cuptr(), current_n);
        current_n = (current_n + GROUP_SIZE - 1) / GROUP_SIZE;
        current_workSize = gpu::WorkSize(GROUP_SIZE, current_n);
        if (is_even_iteration) {
            current_input = reduction_buffer1_gpu;
            current_output = reduction_buffer2_gpu;
        } else {
            current_input = reduction_buffer2_gpu;
            current_output = reduction_buffer1_gpu;
        }
        is_even_iteration = 1 - is_even_iteration;
    }
    
    current_input.copyTo(sum, sizeof(unsigned int));

    CUDA_CHECK_KERNEL(stream);
    
    
}
} // namespace cuda
