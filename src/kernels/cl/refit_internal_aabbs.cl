#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

static inline AABBGPU load_aabb_coherent(volatile __global const BVHNodeGPU* node)
{
    AABBGPU a;
    a.min_x = node->aabb.min_x;
    a.min_y = node->aabb.min_y;
    a.min_z = node->aabb.min_z;
    a.max_x = node->aabb.max_x;
    a.max_y = node->aabb.max_y;
    a.max_z = node->aabb.max_z;
    return a;
}

__kernel void refit_internal_aabbs(
    __global BVHNodeGPU* nodes,
    __global const uint* parent,
    volatile __global uint* arrivalCount,
    uint nfaces)
{
    const uint leafPos = get_global_id(0);
    if (leafPos >= nfaces)
        return;

    uint node = (nfaces - 1u) + leafPos;

    for (;;) {
        const uint p = parent[node];

        if (!atomic_inc(&arrivalCount[p]))
            return;

        mem_fence(CLK_GLOBAL_MEM_FENCE);

        const uint left = nodes[p].leftChildIndex;
        const uint right = nodes[p].rightChildIndex;

        const AABBGPU a = load_aabb_coherent(&nodes[left]);
        const AABBGPU b = load_aabb_coherent(&nodes[right]);

        AABBGPU m;
        m.min_x = fmin(a.min_x, b.min_x);
        m.min_y = fmin(a.min_y, b.min_y);
        m.min_z = fmin(a.min_z, b.min_z);
        m.max_x = fmax(a.max_x, b.max_x);
        m.max_y = fmax(a.max_y, b.max_y);
        m.max_z = fmax(a.max_z, b.max_z);

        nodes[p].aabb = m;

        mem_fence(CLK_GLOBAL_MEM_FENCE);

        if (p == 0u)
            return;
        node = p;
    }
}