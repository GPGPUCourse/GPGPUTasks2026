#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_init_leaves(
    __global const float* vertices,
    __global const uint* faces,
    __global const uint* sorted_triangle_ids,
    __global BVHNodeGPU* nodes,
    __global uint* leaf_triangle_ids,
    __global uint* parent_indices,
    uint nfaces)
{
    const uint i = get_global_id(0);
    if (i >= nfaces) return;

    const uint triangle_id = sorted_triangle_ids[i];
    const uint3 face = loadFace(faces, triangle_id);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    const float3 lo = fmin(a, fmin(b, c));
    const float3 hi = fmax(a, fmax(b, c));
    const uint node_index = nfaces - 1 + i;

    nodes[node_index].aabb.min_x = lo.x;
    nodes[node_index].aabb.min_y = lo.y;
    nodes[node_index].aabb.min_z = lo.z;
    nodes[node_index].aabb.max_x = hi.x;
    nodes[node_index].aabb.max_y = hi.y;
    nodes[node_index].aabb.max_z = hi.z;
    nodes[node_index].leftChildIndex = 0xffffffffu;
    nodes[node_index].rightChildIndex = 0xffffffffu;
    leaf_triangle_ids[i] = triangle_id;

    if (nfaces == 1) parent_indices[0] = 0xffffffffu;
}
