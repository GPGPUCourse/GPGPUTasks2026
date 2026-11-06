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
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    // Лишние потоки тоже участвуют в барьерах, но добавляют ноль.
    local_data[local_index] = index < n ? a[index] : 0;
    __syncthreads();

    // GROUP_SIZE = 256: на каждом шаге число слагаемых уменьшается вдвое.
    for (unsigned int step = GROUP_SIZE / 2; step > 0; step /= 2) {
        if (local_index < step)
            local_data[local_index] += local_data[local_index + step];
        __syncthreads();
    }

    if (local_index == 0)
        atomicAdd(sum, local_data[0]);

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
