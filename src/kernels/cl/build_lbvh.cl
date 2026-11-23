#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline AABBGPU make_bounds(float3 low, float3 high)
{
    AABBGPU bounds = {low.x, low.y, low.z, high.x, high.y, high.z};
    return bounds;
}

static inline AABBGPU empty_bounds(void)
{
    return make_bounds((float3)(INFINITY), (float3)(-INFINITY));
}

static inline AABBGPU union_bounds(AABBGPU a, AABBGPU b)
{
    AABBGPU bounds = {
        fmin(a.min_x, b.min_x), fmin(a.min_y, b.min_y), fmin(a.min_z, b.min_z),
        fmax(a.max_x, b.max_x), fmax(a.max_y, b.max_y), fmax(a.max_z, b.max_z)
    };
    return bounds;
}

static inline void reduce_workgroup_bounds(__local AABBGPU* tile, AABBGPU bounds,
                                           __global AABBGPU* group_bounds)
{
    const uint lid = get_local_id(0);
    tile[lid] = bounds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            tile[lid] = union_bounds(tile[lid], tile[lid + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        group_bounds[get_group_id(0)] = tile[0];
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_triangle_bounds(__global const float* vertices,
                                   __global const uint* faces,
                                   __global float4* centroids,
                                   __global AABBGPU* triangle_bounds,
                                   __global AABBGPU* group_bounds,
                                   uint n)
{
    const uint i = get_global_id(0);
    AABBGPU bounds = empty_bounds();
    if (i < n) {
        const uint3 face = loadFace(faces, i);
        const float3 a = loadVertex(vertices, face.x);
        const float3 b = loadVertex(vertices, face.y);
        const float3 c = loadVertex(vertices, face.z);
        const float3 centroid = (a + b + c) * (1.0f / 3.0f);
        centroids[i] = (float4)(centroid, 0.0f);
        triangle_bounds[i] = make_bounds(fmin(fmin(a, b), c), fmax(fmax(a, b), c));
        bounds = make_bounds(centroid, centroid);
    }

    __local AABBGPU tile[GROUP_SIZE];
    reduce_workgroup_bounds(tile, bounds, group_bounds);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_reduce_bounds(__global const AABBGPU* input,
                                 __global AABBGPU* output, uint n)
{
    const uint i = get_global_id(0);
    __local AABBGPU tile[GROUP_SIZE];
    reduce_workgroup_bounds(tile, i < n ? input[i] : empty_bounds(), output);
}

static inline uint expand_morton_bits(uint x)
{
    x = (x * 0x00010001u) & 0xFF0000FFu;
    x = (x * 0x00000101u) & 0x0F00F00Fu;
    x = (x * 0x00000011u) & 0xC30C30C3u;
    return (x * 0x00000005u) & 0x49249249u;
}

__kernel void lbvh_morton_keys(__global const float4* centroids,
                                __global const AABBGPU* centroid_bounds,
                                __global ulong* keys, uint n)
{
    const uint i = get_global_id(0);
    if (i >= n)
        return;

    const AABBGPU bounds = centroid_bounds[0];
    const float3 low = (float3)(bounds.min_x, bounds.min_y, bounds.min_z);
    const float3 high = (float3)(bounds.max_x, bounds.max_y, bounds.max_z);
    const float3 extent = fmax(high - low, (float3)(1e-9f));
    const float3 normalized = clamp((centroids[i].xyz - low) / extent, 0.0f, 1.0f);
    const uint3 grid = min(convert_uint3_rtz(normalized * 1024.0f), (uint3)(1023u));
    const uint morton = (expand_morton_bits(grid.x) << 2) |
                        (expand_morton_bits(grid.y) << 1) |
                         expand_morton_bits(grid.z);
    keys[i] = ((ulong)morton << 32) | i;
}

// Task06 local bitonic sort and tiled merge path, widened to 64-bit keys.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_sort_tiles(__global const ulong* input, __global ulong* output, uint n)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    __local ulong tile[GROUP_SIZE];
    tile[lid] = gid < n ? input[gid] : ULONG_MAX;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint span = 2; span <= GROUP_SIZE; span <<= 1) {
        for (uint stride = span >> 1; stride > 0; stride >>= 1) {
            if ((lid & stride) == 0) {
                const uint partner = lid ^ stride;
                const ulong a = tile[lid];
                const ulong b = tile[partner];
                const bool ascending = (lid & span) == 0;
                tile[lid] = ascending ? min(a, b) : max(a, b);
                tile[partner] = ascending ? max(a, b) : min(a, b);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (gid < n)
        output[gid] = tile[lid];
}

__kernel void lbvh_merge_tiles(__global const ulong* input, __global ulong* output,
                                uint run_length, uint n)
{
    const uint begin = get_global_id(0) * LBVH_MERGE_ITEMS;
    if (begin >= n)
        return;

    const uint pair_length = 2u * run_length;
    const uint base = begin & ~(pair_length - 1u);
    const uint mid = min(base + run_length, n);
    const uint end = min(base + pair_length, n);
    const uint left_size = mid - base;
    const uint right_size = end - mid;
    const uint diagonal = begin - base;

    uint low = diagonal > right_size ? diagonal - right_size : 0u;
    uint high = min(diagonal, left_size);
    while (low < high) {
        const uint a = low + ((high - low) >> 1);
        const uint b = diagonal - a;
        if (b > 0 && a < left_size && input[mid + b - 1] >= input[base + a])
            low = a + 1;
        else
            high = a;
    }

    uint a = low;
    uint b = diagonal - a;
    const uint stop = min(begin + LBVH_MERGE_ITEMS, end);
    for (uint i = begin; i < stop; ++i) {
        const ulong left = a < left_size ? input[base + a] : ULONG_MAX;
        const ulong right = b < right_size ? input[mid + b] : ULONG_MAX;
        const bool take_left = a < left_size && (b >= right_size || left <= right);
        output[i] = take_left ? left : right;
        a += take_left;
        b += !take_left;
    }
}

static inline int common_prefix(__global const ulong* keys, int n, ulong key, int j)
{
    return j >= 0 && j < n ? (int)clz(key ^ keys[j]) : -1;
}

__kernel void lbvh_build_topology(__global const ulong* keys,
                                  __global const AABBGPU* triangle_bounds,
                                  __global BVHNodeGPU* nodes,
                                  __global uint* leaf_triangles,
                                  __global uchar* prefixes, int n)
{
    const int i = get_global_id(0);
    if (i >= n)
        return;

    const uint leaf_start = n - 1;
    const ulong key = keys[i];
    const uint triangle = (uint)key;
    const BVHNodeGPU leaf = {triangle_bounds[triangle], UINT_MAX, UINT_MAX};
    nodes[leaf_start + i] = leaf;
    leaf_triangles[i] = triangle;
    if (i == n - 1)
        return;

    // Karras range search. The triangle ID makes all keys unique.
    const int previous = common_prefix(keys, n, key, i - 1);
    const int next = common_prefix(keys, n, key, i + 1);
    const int direction = next > previous ? 1 : -1;
    const int outside_prefix = min(previous, next);
    const uint limit = direction > 0 ? n - 1 - i : i;
    uint step = 2;
    while (step <= limit && common_prefix(keys, n, key, i + (int)step * direction) > outside_prefix)
        step <<= 1;

    uint length = 0;
    for (step >>= 1; step > 0; step >>= 1) {
        const uint distance = length + step;
        if (distance <= limit && common_prefix(keys, n, key, i + (int)distance * direction) > outside_prefix)
            length = distance;
    }

    const int j = i + (int)length * direction;
    const int first = min(i, j);
    const int last = max(i, j);
    const ulong first_key = keys[first];
    const int prefix = common_prefix(keys, n, first_key, last);
    int split = first;
    step = last - first;
    do {
        step = (step + 1) >> 1;
        const int candidate = split + step;
        if (candidate < last && common_prefix(keys, n, first_key, candidate) > prefix)
            split = candidate;
    } while (step > 1);

    nodes[i].leftChildIndex = split == first ? leaf_start + split : (uint)split;
    nodes[i].rightChildIndex = split + 1 == last ? leaf_start + split + 1 : (uint)split + 1;
    prefixes[i] = (uchar)prefix;
}

__kernel void lbvh_refit_bounds(__global BVHNodeGPU* nodes,
                                __global const uchar* prefixes,
                                uint leaf_start, uint prefix_begin)
{
    const uint i = get_global_id(0);
    const uint prefix_end = prefix_begin + LBVH_REFIT_PREFIX_BITS;
    if (i >= leaf_start || prefixes[i] < prefix_begin || prefixes[i] >= prefix_end)
        return;

    // Prefix lengths strictly increase down the tree. Each pass reads only leaf
    // boxes or boxes from earlier passes. Inside this band, follow child links
    // until reaching those boxes, without sharing partial results with other work-items.
    uint pending[LBVH_REFIT_PREFIX_BITS + 1];
    uint count = 1;
    pending[0] = i;
    AABBGPU bounds = empty_bounds();
    while (count > 0) {
        const uint node = pending[--count];
        if (node >= leaf_start || prefixes[node] >= prefix_end) {
            bounds = union_bounds(bounds, nodes[node].aabb);
        } else {
            pending[count++] = nodes[node].rightChildIndex;
            pending[count++] = nodes[node].leftChildIndex;
        }
    }
    nodes[i].aabb = bounds;
}
