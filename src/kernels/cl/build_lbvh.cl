#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline AABBGPU empty_bounds()
{
    AABBGPU box;
    box.min_x = box.min_y = box.min_z = INFINITY;
    box.max_x = box.max_y = box.max_z = -INFINITY;
    return box;
}

static inline AABBGPU union_bounds(AABBGPU a, AABBGPU b)
{
    a.min_x = fmin(a.min_x, b.min_x);
    a.min_y = fmin(a.min_y, b.min_y);
    a.min_z = fmin(a.min_z, b.min_z);
    a.max_x = fmax(a.max_x, b.max_x);
    a.max_y = fmax(a.max_y, b.max_y);
    a.max_z = fmax(a.max_z, b.max_z);
    return a;
}

static inline AABBGPU triangle_bounds(__global const float* vertices, __global const uint* faces, uint i)
{
    uint3 face = loadFace(faces, i);
    float3 a = loadVertex(vertices, face.x);
    float3 b = loadVertex(vertices, face.y);
    float3 c = loadVertex(vertices, face.z);
    AABBGPU box;
    box.min_x = fmin(a.x, fmin(b.x, c.x));
    box.min_y = fmin(a.y, fmin(b.y, c.y));
    box.min_z = fmin(a.z, fmin(b.z, c.z));
    box.max_x = fmax(a.x, fmax(b.x, c.x));
    box.max_y = fmax(a.y, fmax(b.y, c.y));
    box.max_z = fmax(a.z, fmax(b.z, c.z));
    return box;
}

static inline float3 triangle_box_center(__global const float* vertices, __global const uint* faces, uint i)
{
    AABBGPU box = triangle_bounds(vertices, faces, i);
    return 0.5f * ((float3)(box.min_x, box.min_y, box.min_z) + (float3)(box.max_x, box.max_y, box.max_z));
}

static inline AABBGPU reduce_local_bounds(AABBGPU box, __local AABBGPU* bounds)
{
    uint tid = get_local_id(0);
    bounds[tid] = box;
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < stride) {
            bounds[tid] = union_bounds(bounds[tid], bounds[tid + stride]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return bounds[0];
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void reduce_centroids(
    __global const float* vertices,
    __global const uint* faces,
    __global AABBGPU* group_bounds,
    uint nfaces)
{
    __local AABBGPU bounds[GROUP_SIZE];
    uint i = get_global_id(0);
    AABBGPU box = empty_bounds();
    if (i < nfaces) {
        float3 centroid = triangle_box_center(vertices, faces, i);
        box.min_x = box.max_x = centroid.x;
        box.min_y = box.max_y = centroid.y;
        box.min_z = box.max_z = centroid.z;
    }
    box = reduce_local_bounds(box, bounds);
    if (get_local_id(0) == 0) {
        group_bounds[get_group_id(0)] = box;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void reduce_bounds(
    __global const AABBGPU* input,
    __global AABBGPU* output,
    uint count)
{
    __local AABBGPU bounds[GROUP_SIZE];
    uint i = get_global_id(0);
    AABBGPU box = empty_bounds();
    if (i < count) {
        box = input[i];
    }
    box = reduce_local_bounds(box, bounds);
    if (get_local_id(0) == 0) {
        output[get_group_id(0)] = box;
    }
}

static inline uint expand_morton_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__kernel void make_morton_codes(
    __global const float* vertices,
    __global const uint* faces,
    __global const AABBGPU* centroid_bounds,
    __global uint* codes,
    __global uint* triangle_ids,
    uint nfaces)
{
    uint i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    float3 centroid = triangle_box_center(vertices, faces, i);
    AABBGPU box = centroid_bounds[0];
    float x = (centroid.x - box.min_x) / fmax(box.max_x - box.min_x, 1e-9f);
    float y = (centroid.y - box.min_y) / fmax(box.max_y - box.min_y, 1e-9f);
    float z = (centroid.z - box.min_z) / fmax(box.max_z - box.min_z, 1e-9f);
    uint ix = min((uint)(clamp(x, 0.0f, 1.0f) * 1024.0f), 1023u);
    uint iy = min((uint)(clamp(y, 0.0f, 1.0f) * 1024.0f), 1023u);
    uint iz = min((uint)(clamp(z, 0.0f, 1.0f) * 1024.0f), 1023u);
    codes[i] = (expand_morton_bits(ix) << 2) | (expand_morton_bits(iy) << 1) | expand_morton_bits(iz);
    triangle_ids[i] = i;
}

__kernel void build_lbvh_leaves(
    __global const float* vertices,
    __global const uint* faces,
    __global const uint* sorted_ids,
    __global BVHNodeGPU* nodes,
    uint nfaces)
{
    uint i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    nodes[nfaces - 1 + i].aabb = triangle_bounds(vertices, faces, sorted_ids[i]);
    nodes[nfaces - 1 + i].leftChildIndex = 0xFFFFFFFFu;
    nodes[nfaces - 1 + i].rightChildIndex = 0xFFFFFFFFu;
}

static inline int lbvh_common_prefix(__global const uint* codes, int nfaces, int i, int j)
{
    if (j < 0 || j >= nfaces) {
        return -1;
    }
    uint diff = codes[i] ^ codes[j];
    if (diff == 0) {
        return 32 + clz((uint)i ^ (uint)j);
    }
    return clz(diff);
}

__kernel void build_lbvh_nodes(
    __global const uint* sorted_codes,
    __global BVHNodeGPU* nodes,
    __global uint* parents,
    __global uint* arrivals,
    uint nfaces)
{
    int i = get_global_id(0);
    if (i >= (int)nfaces - 1) {
        return;
    }
    int cp_left = lbvh_common_prefix(sorted_codes, nfaces, i, i - 1);
    int cp_right = lbvh_common_prefix(sorted_codes, nfaces, i, i + 1);
    int direction = cp_right > cp_left ? 1 : -1;
    int delta_min = lbvh_common_prefix(sorted_codes, nfaces, i, i - direction);
    int length_max = 2;
    while (lbvh_common_prefix(sorted_codes, nfaces, i, i + length_max * direction) > delta_min) {
        length_max <<= 1;
    }
    int length = 0;
    for (int step = length_max >> 1; step > 0; step >>= 1) {
        if (lbvh_common_prefix(sorted_codes, nfaces, i, i + (length + step) * direction) > delta_min) {
            length += step;
        }
    }
    int j = i + length * direction;
    int common_prefix = lbvh_common_prefix(sorted_codes, nfaces, i, j);
    int split_offset = 0;
    for (int step = length; step > 1;) {
        step = (step + 1) >> 1;
        int candidate = split_offset + step;
        if (candidate < length && lbvh_common_prefix(sorted_codes, nfaces, i, i + candidate * direction) > common_prefix) {
            split_offset = candidate;
        }
    }
    int split = i + split_offset * direction + min(direction, 0);
    int first = min(i, j);
    int last = max(i, j);
    uint left = split == first ? nfaces - 1 + split : split;
    uint right = split + 1 == last ? nfaces + split : split + 1;
    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    nodes[i].aabb = empty_bounds();
    parents[left] = i;
    parents[right] = i;
    arrivals[i] = 0;
}

static inline AABBGPU load_published_bounds(volatile __global uint* words)
{
    AABBGPU box;
    box.min_x = as_float(atomic_or(words + 0, 0u));
    box.min_y = as_float(atomic_or(words + 1, 0u));
    box.min_z = as_float(atomic_or(words + 2, 0u));
    box.max_x = as_float(atomic_or(words + 3, 0u));
    box.max_y = as_float(atomic_or(words + 4, 0u));
    box.max_z = as_float(atomic_or(words + 5, 0u));
    return box;
}

static inline void publish_bounds(volatile __global uint* words, AABBGPU box)
{
    atomic_xchg(words + 0, as_uint(box.min_x));
    atomic_xchg(words + 1, as_uint(box.min_y));
    atomic_xchg(words + 2, as_uint(box.min_z));
    atomic_xchg(words + 3, as_uint(box.max_x));
    atomic_xchg(words + 4, as_uint(box.max_y));
    atomic_xchg(words + 5, as_uint(box.max_z));
}

__kernel void propagate_lbvh_bounds(
    __global BVHNodeGPU* nodes,
    __global const uint* parents,
    volatile __global uint* arrivals,
    uint nfaces)
{
    uint i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    uint leaf_start = nfaces - 1;
    uint node = leaf_start + i;
    AABBGPU box = nodes[node].aabb;
    while (node != 0) {
        uint parent = parents[node];
        if (atomic_inc(arrivals + parent) != 1) {
            return;
        }
        read_mem_fence(CLK_GLOBAL_MEM_FENCE);
        uint left = nodes[parent].leftChildIndex;
        uint sibling = node == left ? nodes[parent].rightChildIndex : left;
        AABBGPU other = sibling >= leaf_start
            ? nodes[sibling].aabb
            : load_published_bounds((volatile __global uint*) &nodes[sibling].aabb);
        box = union_bounds(box, other);
        publish_bounds((volatile __global uint*) &nodes[parent].aabb, box);
        write_mem_fence(CLK_GLOBAL_MEM_FENCE);
        node = parent;
    }
}
