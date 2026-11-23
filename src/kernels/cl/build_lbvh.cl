#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline AABBGPU merge_aabb(AABBGPU a, AABBGPU b)
{
    AABBGPU box;
    box.min_x = fmin(a.min_x, b.min_x);
    box.min_y = fmin(a.min_y, b.min_y);
    box.min_z = fmin(a.min_z, b.min_z);
    box.max_x = fmax(a.max_x, b.max_x);
    box.max_y = fmax(a.max_y, b.max_y);
    box.max_z = fmax(a.max_z, b.max_z);

    return box;
}

static inline float3 triangle_centroid(__global const float* vertices,
                                      __global const uint* faces, 
                                      uint i)
{
    uint3 face = loadFace(faces, i);

    return (loadVertex(vertices, face.x) + loadVertex(vertices, face.y)
        + loadVertex(vertices, face.z)) * (1.0f / 3.0f);
}

__kernel void lbvh_centroid_bounds(__global const float* vertices,
                                   __global const uint* faces,
                                   __global AABBGPU* bounds, 
                                   uint n)
{
    uint i = get_global_id(0);
    uint local_id = get_local_id(0);
    __local AABBGPU temp[GROUP_SIZE];

    float3 c = i < n ? triangle_centroid(vertices, faces, i) : (float3)(0.0f);
    AABBGPU box;
    box.min_x = i < n ? c.x : INFINITY;
    box.min_y = i < n ? c.y : INFINITY;
    box.min_z = i < n ? c.z : INFINITY;
    box.max_x = i < n ? c.x : -INFINITY;
    box.max_y = i < n ? c.y : -INFINITY;
    box.max_z = i < n ? c.z : -INFINITY;
    temp[local_id] = box;

    for (uint shift = get_local_size(0) / 2; shift > 0; shift >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < shift) {
            temp[local_id] = merge_aabb(temp[local_id], temp[local_id + shift]);
        }
    }

    if (local_id == 0) {
        bounds[get_group_id(0)] = temp[0];
    }
}

__kernel void lbvh_reduce_bounds(__global const AABBGPU* input,
                                 __global AABBGPU* output, uint n)
{
    uint i = get_global_id(0);
    uint local_id = get_local_id(0);
    __local AABBGPU temp[GROUP_SIZE];

    AABBGPU box;
    box.min_x = box.min_y = box.min_z = INFINITY;
    box.max_x = box.max_y = box.max_z = -INFINITY;
    temp[local_id] = i < n ? input[i] : box;

    for (uint shift = get_local_size(0) / 2; shift > 0; shift >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < shift) {
            temp[local_id] = merge_aabb(temp[local_id], temp[local_id + shift]);
        }
    }

    if (local_id == 0) {
        output[get_group_id(0)] = temp[0];
    }
}

static inline uint expand_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

__kernel void lbvh_morton_codes(__global const float* vertices,
                                __global const uint* faces,
                                __global const AABBGPU* bounds,
                                __global uint* codes, 
                                __global uint* indices, 
                                uint n)
{
    uint i = get_global_id(0);

    if (i < n) {
        AABBGPU box = bounds[0];
        float3 c = triangle_centroid(vertices, faces, i);
        float3 lo = (float3)(box.min_x, box.min_y, box.min_z);
        float3 extent = fmax((float3)(box.max_x, box.max_y, box.max_z) - lo, (float3)(1e-9f));

        float3 normalized = clamp((c - lo) / extent, 0.0f, 1.0f);
        uint3 grid = min(convert_uint3(normalized * 1024.0f), (uint3)(1023u));

        codes[i] = (expand_bits(grid.x) << 2) | (expand_bits(grid.y) << 1) | expand_bits(grid.z);
        indices[i] = i;
    }
}

__kernel void lbvh_init_leaves(__global const float* vertices,
                               __global const uint* faces,
                               __global const uint* indices,
                               __global BVHNodeGPU* nodes,
                               __global uint* flags, 
                               uint n)
{
    uint i = get_global_id(0);

    if (i < n) {
        uint3 face = loadFace(faces, indices[i]);
        float3 a = loadVertex(vertices, face.x);
        float3 b = loadVertex(vertices, face.y);
        float3 c = loadVertex(vertices, face.z);

        float3 lo = fmin(a, fmin(b, c));
        float3 hi = fmax(a, fmax(b, c));

        BVHNodeGPU node;
        node.aabb.min_x = lo.x;
        node.aabb.min_y = lo.y;
        node.aabb.min_z = lo.z;
        node.aabb.max_x = hi.x;
        node.aabb.max_y = hi.y;
        node.aabb.max_z = hi.z;
        node.leftChildIndex = node.rightChildIndex = 0xFFFFFFFFu;
        nodes[n - 1 + i] = node;

        if (i < n - 1) {
            flags[i] = 0;
        }
    }
}

static inline int lbvh_common_prefix(__global const uint* codes, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;

    uint diff = codes[i] ^ codes[j];
    return diff == 0 ? 32 + (int)clz((uint)(i ^ j)) : (int)clz(diff);
}

__kernel void lbvh_build_tree(__global const uint* codes,
                              __global BVHNodeGPU* nodes,
                              __global uint* parents, 
                              uint n)
{
    int i = get_global_id(0);
    if (i >= (int)n - 1) return;

    int direction = lbvh_common_prefix(codes, n, i, i + 1)
        > lbvh_common_prefix(codes, n, i, i - 1) ? 1 : -1;
    int delta_min = lbvh_common_prefix(codes, n, i, i - direction);

    int max_length = 2;
    while (lbvh_common_prefix(codes, n, i, i + max_length * direction) > delta_min) {
        max_length <<= 1;
    }

    int length = 0;
    for (int step = max_length / 2; step > 0; step >>= 1) {
        if (lbvh_common_prefix(codes, n, i, i + (length + step) * direction) > delta_min) {
            length += step;
        }
    }

    int j = i + length * direction;
    int first = min(i, j);
    int last = max(i, j);

    int common = lbvh_common_prefix(codes, n, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        int candidate = split + step;
        if (candidate < last && lbvh_common_prefix(codes, n, first, candidate) > common) {
            split = candidate;
        }
    } while (step > 1);

    uint left = split == first ? n - 1 + split : split;
    uint right = split + 1 == last ? n - 1 + split + 1 : split + 1;

    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;

    parents[left] = i;
    parents[right] = i;
}

__kernel void lbvh_propagate_aabb(__global BVHNodeGPU* nodes,
                                  __global const uint* parents,
                                  volatile __global uint* flags, 
                                  uint n)
{
    uint i = get_global_id(0);

    if (i >= n) return;

    uint node = n - 1 + i;

    while (node != 0) {
        uint parent = parents[node];
        mem_fence(CLK_GLOBAL_MEM_FENCE);

        if (atomic_inc(flags + parent) == 0) return;

        mem_fence(CLK_GLOBAL_MEM_FENCE);
        nodes[parent].aabb = merge_aabb(nodes[nodes[parent].leftChildIndex].aabb,
                                        nodes[nodes[parent].rightChildIndex].aabb);

        node = parent;
    }
}
