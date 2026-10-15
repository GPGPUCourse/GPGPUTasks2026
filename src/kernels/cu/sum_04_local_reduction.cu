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
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    const uint warp_id = local_index >> 5;
    const uint lilip_id = local_index % 32;

    uint val = (index < n) ? a[index] : 0;

    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }

    __shared__ unsigned int local_data[GROUP_SIZE / 32];
    if (lilip_id == 0) {
        local_data[warp_id] = val;
    }
    __syncthreads();

    // TODO
    for (uint i = (GROUP_SIZE >> 5) >> 1; i > 0; i >>= 1) {
        if (warp_id < i && lilip_id == 0) {
            local_data[warp_id] += local_data[warp_id + i];
        }
        __syncthreads();
    }

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
