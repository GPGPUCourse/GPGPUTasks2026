#include "../shared_structs/bvh_node_gpu_shared.h"
#include "aabb_helpers.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>

__global__ void build_lbvh_bounds(
    BVHNodeGPU* nodes,
    const unsigned int* parents,
    unsigned int* ready,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    unsigned int current = n - 1 + i;

    while (current != 0) {
        unsigned int parentIndex = parents[current];

        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            counter(ready[parentIndex]);

        unsigned int previous =
            counter.fetch_add(1u, cuda::memory_order_acq_rel);

        if (previous == 0)
            return;

        BVHNodeGPU& parent = nodes[parentIndex];

        parent.aabb = merge_aabb(
            nodes[parent.leftChildIndex].aabb,
            nodes[parent.rightChildIndex].aabb);

        current = parentIndex;
    }
}