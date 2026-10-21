#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    unsigned int* tree_sum,
    unsigned int n,
    unsigned int glob_stride)
{
    unsigned loc_id = threadIdx.x;
    unsigned glob_id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned idx = (glob_id + 1) * glob_stride - 1;

    __shared__ unsigned loc_data[GROUP_SIZE];

    loc_data[loc_id] = idx < n ? tree_sum[idx] : 0;
    __syncthreads();

    for (unsigned loc_stride = 2; loc_stride <= GROUP_SIZE; loc_stride <<= 1) {
        if ((loc_id + 1) % loc_stride == 0)
            loc_data[loc_id] += loc_data[loc_id - loc_stride / 2];
        __syncthreads();
    }

    if (idx < n)
        tree_sum[idx] = loc_data[loc_id];
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize& workSize, gpu::gpu_mem_32u& tree_sum, unsigned int n, unsigned int glob_stride)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(tree_sum.cuptr(), n, glob_stride);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
