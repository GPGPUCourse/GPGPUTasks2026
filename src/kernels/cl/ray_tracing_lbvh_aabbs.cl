#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "aabb_helpers.cl"

__kernel void ray_tracing_lbvh_aabbs(
    __global const AABBGPU* leaf_aabbs, // default
    __global const uint* leafTriIndices,// sorted
    __global const uint* parentsIndices,// sorted
    __global uint* checkpointCounters,  // sorted
    __global BVHNodeGPU* nodes,         // sorted
    uint n
) {
    const uint gid = get_global_id(0); // sorted

    if (n == 0 || gid >= n) {
        return;
    }

    if (n == 1) {
        if (gid == 0) {
            nodes[0].aabb = leaf_aabbs[leafTriIndices[0]];
        }
        return;
    }

    size_t leafIndex = (n - 1) + gid; // sorted

    nodes[leafIndex].aabb = leaf_aabbs[leafTriIndices[gid]]; // ok

    uint node = leafIndex; // sorted
    uint parent = parentsIndices[leafIndex]; // sorted
    while (parent != UINT_MAX) {
        uint old = atomic_add(checkpointCounters + parent, 1);
        if (old == 0) {
            break;
        }
        nodes[parent].aabb = merge_aabb(nodes[nodes[parent].leftChildIndex].aabb, nodes[nodes[parent].rightChildIndex].aabb);
        parent = parentsIndices[parent];
    }
}