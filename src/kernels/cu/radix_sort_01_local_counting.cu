#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

struct GroupCounts {
  __host__ __device__ uint32_t &operator[](uint32_t i) { return data[i]; }
  __host__ __device__ uint32_t operator[](uint32_t i) const { return data[i]; }
  uint32_t data[NGROUPS];
};

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* elems,
          GroupCounts* counts,
          unsigned int bit,
          unsigned int n)
{
    __shared__ GroupCounts groups_cnt;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t elem = (idx < n) ? elems[idx] : (-1u);
    uint32_t mask = (1 << ITER_BITS) - 1;
    uint32_t group = (elem >> bit) & mask; 
    if (threadIdx.x == 0) {
      for (int i = 0; i < NGROUPS; ++i) {
        groups_cnt[i] = 0;
      }
    }
    __syncthreads();
    atomicAdd(groups_cnt.data + group, 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
      counts[blockIdx.x] = groups_cnt;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), (GroupCounts*)buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
