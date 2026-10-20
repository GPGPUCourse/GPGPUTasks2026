#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define BUF_SIZE (GROUP_SIZE / WARP_SIZE)

__global__ void prefix_one_group_reduction(
    unsigned int* pow2_sum,
    unsigned int from,
    unsigned int to)
{
    __shared__ unsigned int buf[BUF_SIZE];
    const unsigned int lid = threadIdx.x;
    const unsigned int lane = lid & (WARP_SIZE - 1);

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sum = pow2_sum[x];
    // warp-wide reduce (sub_group_reduce_add): butterfly через __shfl_down_sync
    // на V100 (SM 7.0) __reduce_add_sync недоступен (появился только в SM 8.0)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    x >>= WARP_SHIFT;
    if (lane == 0) {
        pow2_sum[to + x] = buf[lid >> WARP_SHIFT] = sum;
    }

    __syncthreads();

    if (lid == 0) {
        unsigned int sum = buf[0];
        for (int i = 1; i < BUF_SIZE; ++i) {
            sum += buf[i];
            buf[i] = sum;
        }
    }
}

namespace cuda {
void prefix_one_group_reduction(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &pow2_sum, unsigned int from, unsigned int to)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_one_group_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), from, to);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
