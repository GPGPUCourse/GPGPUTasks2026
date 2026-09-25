#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    // Подсказки:
    const uint32_t offset = blockIdx.x * blockDim.x * LOAD_K_VALUES_PER_ITEM + threadIdx.x;
    const uint32_t local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    uint32_t threadValue = 0;
    for (uint32_t i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
      if (offset + GROUP_SIZE * i < n) {
        threadValue += a[offset + GROUP_SIZE * i];
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
    
    __syncthreads();
    if (local_index == 0) {
      b[blockIdx.x] = local_data[0];
    }
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
