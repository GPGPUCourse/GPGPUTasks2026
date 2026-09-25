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
    const uint32_t index = blockIdx.x * blockDim.x * LOAD_K_VALUES_PER_ITEM + threadIdx.x;
    const uint32_t local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    uint32_t threadValue = 0;
    for (uint32_t i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
      if (index + GROUP_SIZE * i < n) {
        threadValue += a[index + GROUP_SIZE * i];
      }
    }
    local_data[local_index] = threadValue;
    
    uint32_t curThreads = GROUP_SIZE >> 1;
    while(curThreads > 0) {
      __syncthreads();
      if (local_index < curThreads) {
        local_data[local_index] += local_data[local_index + curThreads];
      }
      curThreads >>= 1;
    }

    if (local_index == 0) {
      atomicAdd(sum, local_data[0]);
    }
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
