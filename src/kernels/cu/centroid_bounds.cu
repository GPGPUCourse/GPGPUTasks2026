#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "aabb_helpers.cuh"
#include <float.h>
#include <type_traits>

template <bool IsBoxes>
__global__ void reduce_bounds_blocks(
    const std::conditional_t<IsBoxes, AABBGPU, float>* input,
    AABBGPU* blockBounds,
    int n)
{
    __shared__ AABBGPU partial[GROUP_SIZE];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    AABBGPU& value = partial[t];
    value.min_x = value.min_y = value.min_z = FLT_MAX;
    value.max_x = value.max_y = value.max_z = -FLT_MAX;

    if (i < n) {
        if constexpr (IsBoxes) {
            value = input[i];
        } else {
            value.min_x = value.max_x = input[3 * i];
            value.min_y = value.max_y = input[3 * i + 1];
            value.min_z = value.max_z = input[3 * i + 2];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (t < stride) {
            AABBGPU& a = partial[t];
            const AABBGPU& b = partial[t + stride];

            a = merge_aabb(a, b);
        }

        __syncthreads();
    }

    if (t == 0)
        blockBounds[blockIdx.x] = partial[0];
}

namespace cuda {

gpu::shared_device_buffer_typed<AABBGPU> centroid_bounds(
    const gpu::gpu_mem_32f& centroids,
    int n)
{
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();

    int count = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    gpu::shared_device_buffer_typed<AABBGPU> current(count);
    gpu::shared_device_buffer_typed<AABBGPU> next(count);

    gpu::WorkSize firstWorkSize(GROUP_SIZE, n);

    ::reduce_bounds_blocks<false>
        <<<firstWorkSize.cuGridSize(), firstWorkSize.cuBlockSize(), 0, stream>>>(
            centroids.cuptr(), current.cuptr(), n);

    while (count > 1) {
        gpu::WorkSize workSize(GROUP_SIZE, count);

        ::reduce_bounds_blocks<true>
            <<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
                current.cuptr(), next.cuptr(), count);

        count = (count + GROUP_SIZE - 1) / GROUP_SIZE;
        current.swap(next);
    }

    CUDA_CHECK_KERNEL_SYNC(stream);
    return current;
}

}