#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <cuda/atomic>

#include "../defines.h"

__global__ void sum_03_local_memory_atomic_per_workgroup(
    const unsigned int* a,
    unsigned int* sum,
    unsigned int  n)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t local_idx = threadIdx.x;

    __shared__ int shmem[GROUP_SIZE];

    shmem[local_idx] = (idx < n) ? a[idx] : 0;
    __syncthreads();

    if (local_idx == 0) {
        int local_sum = 0;
        for (int id = 0; id < GROUP_SIZE; id++) {
            local_sum += shmem[id];
        }

        atomicAdd(sum, local_sum);
        //                |
        // very very bad \|/
        //                v
        // cuda::atomic_ref<unsigned int> ref(*sum);
        // ref.fetch_add(local_sum);
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
