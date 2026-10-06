#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    unsigned int* a,
    unsigned int* b,
    unsigned int* sum,
    unsigned int  n)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    const uint group_id = blockIdx.x;

    __shared__ unsigned int local_data[GROUP_SIZE];

    local_data[local_index] = (index < n) ? a[index] : 0;
    __syncthreads();

    if (local_index == 0) {
        unsigned int total_sum = 0;
        for (unsigned int i = 0; i < blockDim.x; i++) {
            total_sum += local_data[i];
        }

        if (blockIdx.x == 0 && gridDim.x == 1) {
            *sum = total_sum;
        } else {
            b[group_id] = total_sum;
        }
    }
}

namespace cuda {
void sum_04_local_reduction(gpu::WorkSize &workSize,
    gpu::gpu_mem_32u &input,
    gpu::gpu_mem_32u &buf1,
    gpu::gpu_mem_32u &buf2,
    gpu::gpu_mem_32u &sum,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();

    gpu::gpu_mem_32u* a = &input;
    gpu::gpu_mem_32u* b = &buf1;

    while (n > GROUP_SIZE) {
        unsigned int grid_size = div_ceil(n, (unsigned int)GROUP_SIZE);
        ::sum_04_local_reduction<<<grid_size, GROUP_SIZE, 0, stream>>>(a->cuptr(), b->cuptr(), sum.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
        a = b;
        b = (b == &buf1) ? &buf2 : &buf1;
        n = grid_size;
    }
    ::sum_04_local_reduction<<<1, GROUP_SIZE, 0, stream>>>(a->cuptr(), b->cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
