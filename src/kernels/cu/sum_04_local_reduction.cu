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
    __shared__ unsigned int local_data[GROUP_SIZE/WARP_SIZE];

    uint32_t threadValue = 0;
    for (uint32_t i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
      if (offset + GROUP_SIZE * i < n) {
        threadValue += a[offset + GROUP_SIZE * i];
      }
    }

    threadValue += __shfl_xor_sync(0xffffffff, threadValue, 16);
    threadValue += __shfl_xor_sync(0xffffffff, threadValue, 8);
    threadValue += __shfl_xor_sync(0xffffffff, threadValue, 4);
    threadValue += __shfl_xor_sync(0xffffffff, threadValue, 2);
    threadValue += __shfl_xor_sync(0xffffffff, threadValue, 1);
    if (local_index % WARP_SIZE == 0) {
      local_data[local_index / WARP_SIZE] = threadValue;
    }
    
    __syncthreads();
    if (local_index == 0) {
      uint32_t res = 0;
      for (int i = 0; i < GROUP_SIZE/WARP_SIZE; ++i) {
        res += local_data[i];
      }
      b[blockIdx.x] = res;
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
