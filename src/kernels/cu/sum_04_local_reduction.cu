#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <cuda/atomic>

#include <vector>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{


    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;

    __shared__ unsigned int local_data[GROUP_SIZE];
    local_data[local_index] = (index < n) ? a[index] : 0;

    __syncthreads();

    for (int step = GROUP_SIZE / 2; step >= WARP_SIZE; step /= 2) {
        if (local_index < step) {
            local_data[local_index] += local_data[local_index + step];
        }

        __syncthreads();
    }

    if (local_index < 32) {
        uint val = local_data[local_index];

        for (uint offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }

        if (local_index == 0) {
            b[blockIdx.x] = val;
        }
    }
}
namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n,
    gpu::gpu_mem_32u &reduction_buffer1_gpu, gpu::gpu_mem_32u &reduction_buffer2_gpu)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), reduction_buffer1_gpu.cuptr(), n);
    
    for (int size = div_ceil(n, (unsigned int)GROUP_SIZE); size > 1; size = (size + GROUP_SIZE - 1) / GROUP_SIZE) {
        gpu::WorkSize workSize(GROUP_SIZE, size);
        ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(reduction_buffer1_gpu.cuptr(), reduction_buffer2_gpu.cuptr(), size);
        reduction_buffer1_gpu.swap(reduction_buffer2_gpu);
    }

    reduction_buffer1_gpu.copyToN(sum, 1);

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
