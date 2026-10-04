#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void sum_03_local_memory_atomic_per_workgroup(
    const unsigned int* a,
    unsigned int* sum,
    unsigned int  n)
{
    // Подсказки:
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    unsigned int my_sum = 0;

    for (unsigned int i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        if (index < n / LOAD_K_VALUES_PER_ITEM) {
            my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
        }
    }
    local_data[local_index] = my_sum;
    __syncthreads();
    
    uint block_sum = 0;
    if (local_index == 0) {
        
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            block_sum += local_data[i];
            
        }

        atomicAdd(sum, block_sum);
    }

    

    // TODO
}

namespace cuda {
void sum_03_local_memory_atomic_per_workgroup(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_03_local_memory_atomic_per_workgroup<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
