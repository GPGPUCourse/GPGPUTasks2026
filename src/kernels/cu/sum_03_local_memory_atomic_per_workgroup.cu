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
    __shared__ unsigned int smem[GROUP_SIZE];
    const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    if (index < n) {
        smem[tid] = a[index];
    } else {
        smem[tid] = 0;
    }

    __syncthreads();
    if (tid == 0) {
        unsigned int group_sum = 0;
        for (unsigned int k = 0; k < GROUP_SIZE; ++k) {
            group_sum += smem[k];
        }
        atomicAdd(sum, group_sum);
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
