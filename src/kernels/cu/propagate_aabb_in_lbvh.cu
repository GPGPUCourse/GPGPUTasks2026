#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/bvh_node_gpu_shared.h"

__global__ void propagate_aabb_in_lbvh(
    BVHNodeGPU* outNodes,
    const unsigned int* parents,
    unsigned int* flags,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    unsigned int nodeIndex = (nfaces - 1) + i;

    while (nodeIndex != 0) {
        const unsigned int parentIndex = parents[nodeIndex];

        if (atomicInc(&flags[parentIndex], 1) == 1) {
            return;
        }

        BVHNodeGPU& node = outNodes[parentIndex];
        const BVHNodeGPU& left = outNodes[node.leftChildIndex];
        const BVHNodeGPU& right = outNodes[node.rightChildIndex];

        AABBGPU aabb {};
        aabb.min_x = min(left.aabb.min_x, right.aabb.min_x);
        aabb.min_y = min(left.aabb.min_y, right.aabb.min_y);
        aabb.min_z = fminf(left.aabb.min_z, right.aabb.min_z);
        aabb.max_x = max(left.aabb.max_x, right.aabb.max_x);
        aabb.max_y = max(left.aabb.max_y, right.aabb.max_y);
        aabb.max_z = max(left.aabb.max_z, right.aabb.max_z);

        node.aabb = aabb;
        nodeIndex = parentIndex;
    }
}

namespace cuda {

void propagate_aabb_in_lbvh(
    const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& outNodes,
    const gpu::gpu_mem_32u& parents,
    gpu::gpu_mem_32u& flags,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::propagate_aabb_in_lbvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        outNodes.cuptr(),
        parents.cuptr(),
        flags.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
