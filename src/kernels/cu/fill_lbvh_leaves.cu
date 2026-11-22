#include "../shared_structs/bvh_node_gpu_shared.h"

#include <climits>
#include <cuda_runtime.h>

__global__ void fill_lbvh_leaves(
    const AABBGPU* triangleBoxes,
    const unsigned int* triangleIndices,
    BVHNodeGPU* nodes,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    unsigned int triangleId = triangleIndices[i];
    int leafIndex = n - 1 + i;

    BVHNodeGPU& leaf = nodes[leafIndex];
    leaf.aabb = triangleBoxes[triangleId];
    leaf.leftChildIndex = UINT_MAX;
    leaf.rightChildIndex = UINT_MAX;
}