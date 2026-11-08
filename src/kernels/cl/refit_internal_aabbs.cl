#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

// Node layout assumed by the vector accesses below (32 bytes, checked by static_assert on the host):
//   floats 0..5 : min_x min_y min_z max_x max_y max_z
//   uints  6..7 : leftChildIndex rightChildIndex
// So a node is two float4s: [min.xyz, max.x] and [max.y, max.z, left, right].
#define NODE_FLOAT4S 2u

typedef struct {
    float3 mn;
    float3 mx;
} Box;

static inline Box box_from_halves(float4 lo, float4 hi)
{
    Box b;
    b.mn = lo.xyz;
    b.mx = (float3)(lo.w, hi.x, hi.y);
    return b;
}

// Plain read: only for data that no other work-item is writing in this kernel (our own leaf).
static inline Box load_box(__global const BVHNodeGPU* nodes, uint index)
{
    __global const float4* n = (__global const float4*)nodes + index * NODE_FLOAT4S;
    return box_from_halves(n[0], n[1]);
}

// Volatile read: for a sibling whose AABB was just published by another work-item.
static inline Box load_box_coherent(__global const BVHNodeGPU* nodes, uint index)
{
    volatile __global const float4* n = (volatile __global const float4*)nodes + index * NODE_FLOAT4S;
    return box_from_halves(n[0], n[1]);
}

// Writes only the 24 AABB bytes; the child indices in bytes 24..31 are left untouched.
static inline void store_box(__global BVHNodeGPU* nodes, uint index, Box b)
{
    __global float* f = (__global float*)nodes + index * 8u;
    vstore4((float4)(b.mn, b.mx.x), 0, f);
    vstore2((float2)(b.mx.y, b.mx.z), 0, f + 4);
}

static inline Box merge_box(Box a, Box b)
{
    Box m;
    m.mn = fmin(a.mn, b.mn);
    m.mx = fmax(a.mx, b.mx);
    return m;
}

// One work-item per leaf. It walks up toward the root; at each parent the first child to arrive
// stops, and the second child merges both AABBs and continues. up[node] = (parent, sibling).
__kernel void refit_internal_aabbs(
    __global BVHNodeGPU* nodes,
    __global const uint2* up,
    volatile __global uint* arrival_count,
    uint nfaces)
{
    const uint leaf_pos = get_global_id(0);
    if (leaf_pos >= nfaces)
        return;

    uint node = (nfaces - 1u) + leaf_pos;
    Box box = load_box(nodes, node); // our own AABB stays in registers from now on
    uint2 link = up[node];

    for (;;) {
        const uint parent = link.x;
        const uint sibling = link.y;

        const uint2 p = up[parent];

        if (!atomic_inc(&arrival_count[parent]))
            return;

        mem_fence(CLK_GLOBAL_MEM_FENCE);

        box = merge_box(box, load_box_coherent(nodes, sibling));
        store_box(nodes, parent, box);

        mem_fence(CLK_GLOBAL_MEM_FENCE);

        if (parent == 0u)
            return;
        link = p;
    }
}