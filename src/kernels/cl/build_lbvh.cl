#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "geometry_helpers.cl"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"


static inline uint expand_bits(uint value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;

    return value;
}


static inline int common_prefix(
    __global const uint* codes,
    int n,
    int i,
    int j
) {
    if (j < 0 || j >= n) return -1;
    const uint a = codes[i];
    const uint b = codes[j];
    return a == b ? 32 + clz((uint)(i ^ j)) : clz(a ^ b);
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void clean_counters(
    __global uint*  counters,
             uint   n
) {
    const uint id = get_global_id(0);
    if (id < n) counters[id] = 0;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void make_morton_codes(
    __global const float* vertices,
    __global const uint*  faces,
    __global       uint*  codes,
    __global       uint*  indices,
                   uint   nfaces,
                   float min_point_x, float min_point_y, float min_point_z,
                   float max_point_x, float max_point_y, float max_point_z
) {
    const uint id = get_global_id(0);
    if (id >= nfaces) return;

    indices[id] = id;

    uint3 face = loadFace(faces, id);
    float3 a = loadVertex(vertices, face.x);
    float3 b = loadVertex(vertices, face.y);
    float3 c = loadVertex(vertices, face.z);
    float3 centroid = (a + b + c) / 3.0f;
    float length_x = max_point_x - min_point_x;
    float length_y = max_point_y - min_point_y;
    float length_z = max_point_z - min_point_z;

    const uint xx = expand_bits((uint)(clamp((centroid.x - min_point_x) / fmax(length_x, 1e-9f), 0.0f, 1.0f) * 1023.0f));
    const uint yy = expand_bits((uint)(clamp((centroid.y - min_point_y) / fmax(length_y, 1e-9f), 0.0f, 1.0f) * 1023.0f));
    const uint zz = expand_bits((uint)(clamp((centroid.z - min_point_z) / fmax(length_z, 1e-9f), 0.0f, 1.0f) * 1023.0f));
    codes[id] = xx * 4u + yy * 2u + zz;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sort_codes(
    __global uint* codes,
    __global uint* indices,
             uint padding,
             uint width,
             uint stride
) {
    uint id = get_global_id(0);
    if (id >= padding)
        return;

    uint pair = id ^ stride;
    if (pair < id)
        return;

    bool leftGreater = codes[id] > codes[pair] || (codes[id] == codes[pair] && indices[id] > indices[pair]);
    bool ascending = (id & width) == 0;
    if (leftGreater != ascending)
        return;

    uint code = codes[id];
    codes[id] = codes[pair];
    codes[pair] = code;

    uint index = indices[id];
    indices[id] = indices[pair];
    indices[pair] = index;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void build_tree_structure(
    __global const uint*       sorted_codes,
    __global       BVHNodeGPU* nodes,
    __global       uint*       parents,
                   uint        n
) {
    const int id = (int)get_global_id(0);
    if (id >= (int)n - 1) return;

    const int l = common_prefix(sorted_codes, (int)n, id, id - 1);
    const int r = common_prefix(sorted_codes, (int)n, id, id + 1);
    const int d = r > l ? 1 : -1;
    const int min_prefix = common_prefix(sorted_codes, (int)n, id, id - d);
    int max_length = 2;
    while (common_prefix(sorted_codes, (int)n, id, id + max_length * d) > min_prefix) {
        max_length *= 2;
    }
    int length = 0;
    for (int step = max_length / 2; step > 0; step /= 2) {
        if (common_prefix(sorted_codes, (int)n, id, id + (length + step) * d) > min_prefix) {
            length += step;
        }
    }
    const int other = id + length * d;
    const int first = min(id, other);
    const int last = max(id, other);
    const int range_prefix = common_prefix(sorted_codes, (int)n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) / 2;
        const int candidate = split + step;
        if (candidate < last && common_prefix(sorted_codes, (int)n, first, candidate) > range_prefix) {
            split = candidate;
        }
    } while (step > 1);

    const uint left = split == first ? n - 1u + (uint)split : (uint)split;
    const uint right = split + 1 == last ? n + (uint)split : (uint)(split + 1);
    nodes[id].leftChildIndex = left;
    nodes[id].rightChildIndex = right;
    parents[left] = (uint)id;
    parents[right] = (uint)id;
    if (id == 0) parents[0] = UINT_MAX;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void init_leaves(
    __global const float*      vertices,
    __global const uint*       faces,
    __global const uint*       sorted_triangle_indices,
    __global       BVHNodeGPU* nodes,
                   uint        n
) {
    const uint id = get_global_id(0);
    if (id >= n) return;

    const uint3 face = loadFace(faces, sorted_triangle_indices[id]);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    const uint node_index = n - 1u + id;

    nodes[node_index].aabb.min_x = fmin(a.x, fmin(b.x, c.x));
    nodes[node_index].aabb.min_y = fmin(a.y, fmin(b.y, c.y));
    nodes[node_index].aabb.min_z = fmin(a.z, fmin(b.z, c.z));
    nodes[node_index].aabb.max_x = fmax(a.x, fmax(b.x, c.x));
    nodes[node_index].aabb.max_y = fmax(a.y, fmax(b.y, c.y));
    nodes[node_index].aabb.max_z = fmax(a.z, fmax(b.z, c.z));

    nodes[node_index].leftChildIndex = UINT_MAX;
    nodes[node_index].rightChildIndex = UINT_MAX;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void compute_aabbs(
             __global const uint*       parents,
    volatile __global       BVHNodeGPU* nodes,
    volatile __global       uint*       counters,
                            uint        n
) {
    const uint leaf = get_global_id(0);
    if (leaf >= n || n <= 1u) return;
    uint current = n - 1u + leaf;

    while (parents[current] != UINT_MAX) {
        const uint parent = parents[current];
        mem_fence(CLK_LOCAL_MEM_FENCE);
        if (atomic_inc(&counters[parent]) == 0u) return;

        const BVHNodeGPU left = nodes[nodes[parent].leftChildIndex];
        const BVHNodeGPU right = nodes[nodes[parent].rightChildIndex];
        nodes[parent].aabb.min_x = fmin(left.aabb.min_x, right.aabb.min_x);
        nodes[parent].aabb.min_y = fmin(left.aabb.min_y, right.aabb.min_y);
        nodes[parent].aabb.min_z = fmin(left.aabb.min_z, right.aabb.min_z);
        nodes[parent].aabb.max_x = fmax(left.aabb.max_x, right.aabb.max_x);
        nodes[parent].aabb.max_y = fmax(left.aabb.max_y, right.aabb.max_y);
        nodes[parent].aabb.max_z = fmax(left.aabb.max_z, right.aabb.max_z);

        mem_fence(CLK_LOCAL_MEM_FENCE);
        current = parent;
    }
}
