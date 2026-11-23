#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline AABBGPU make_box(float3 lo, float3 hi)
{
    return (AABBGPU){lo.x, lo.y, lo.z, hi.x, hi.y, hi.z};
}

static inline AABBGPU empty_box()
{
    return make_box((float3)(INFINITY), (float3)(-INFINITY));
}

static inline AABBGPU merge_boxes(AABBGPU a, AABBGPU b)
{
    return (AABBGPU){
        fmin(a.min_x, b.min_x), fmin(a.min_y, b.min_y), fmin(a.min_z, b.min_z),
        fmax(a.max_x, b.max_x), fmax(a.max_y, b.max_y), fmax(a.max_z, b.max_z)
    };
}

static inline float3 centroid(__global const float* vertices, __global const uint* faces, uint i)
{
    const uint3 f = loadFace(faces, i);
    return (loadVertex(vertices, f.x) + loadVertex(vertices, f.y) + loadVertex(vertices, f.z)) * (1.0f / 3.0f);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_centroid_bounds(__global const float* vertices, __global const uint* faces,
                                   __global AABBGPU* bounds, uint offset, uint count)
{
    const uint i = get_global_id(0);
    const uint lane = get_local_id(0);
    __local float4 lo[GROUP_SIZE];
    __local float4 hi[GROUP_SIZE];

    const float3 c = i < count ? centroid(vertices, faces, i) : (float3)(0.0f);
    lo[lane] = i < count ? (float4)(c, 0.0f) : (float4)(INFINITY);
    hi[lane] = i < count ? (float4)(c, 0.0f) : (float4)(-INFINITY);
    for (uint step = GROUP_SIZE / 2; step > 0; step /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < step) {
            lo[lane] = fmin(lo[lane], lo[lane + step]);
            hi[lane] = fmax(hi[lane], hi[lane + step]);
        }
    }
    if (lane == 0)
        bounds[offset + get_group_id(0)] = make_box(lo[0].xyz, hi[0].xyz);
}

// A dispatch writes one level of the binary tree and reads only the level below it.
__kernel void lbvh_reduce_boxes(__global AABBGPU* bounds, uint level_size)
{
    const uint i = get_global_id(0);
    if (i < level_size) {
        const uint node = level_size + i;
        bounds[node] = merge_boxes(bounds[2 * node], bounds[2 * node + 1]);
    }
}

static inline uint expand_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    return (v * 0x00000005u) & 0x49249249u;
}

__kernel void lbvh_morton_codes(__global const float* vertices, __global const uint* faces,
                                __global const AABBGPU* bounds, __global uint2* keys, uint count)
{
    const uint i = get_global_id(0);
    if (i >= count)
        return;

    const AABBGPU box = bounds[1];
    const float3 lo = (float3)(box.min_x, box.min_y, box.min_z);
    const float3 hi = (float3)(box.max_x, box.max_y, box.max_z);
    const float3 p = (centroid(vertices, faces, i) - lo) / fmax(hi - lo, (float3)(1e-9f));
    const uint3 q = convert_uint3(clamp(p * 1024.0f, 0.0f, 1023.0f));
    const uint code = (expand_bits(q.x) << 2) | (expand_bits(q.y) << 1) | expand_bits(q.z);
    keys[i] = (uint2)(code, i);
}

// Stable parallel merge as in Task06, with the original face ID carried along.
__kernel void lbvh_merge_keys(__global const uint2* input, __global uint2* output,
                               uint run_size, uint count)
{
    const uint i = get_global_id(0);
    if (i >= count)
        return;

    const uint start = i & ~(2 * run_size - 1);
    const uint middle = min(start + run_size, count);
    const uint end = min(start + 2 * run_size, count);
    const bool left = i < middle;
    const uint2 key = input[i];

    uint lo = left ? middle : start;
    uint hi = left ? end : middle;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        const uint other = input[mid].x;
        if (other < key.x || (!left && other == key.x))
            lo = mid + 1;
        else
            hi = mid;
    }
    output[i + lo - middle] = key;
}

__kernel void lbvh_make_leaves(__global const float* vertices, __global const uint* faces,
                                __global const uint2* keys, __global BVHNodeGPU* nodes,
                                __global uint* leaf_faces, uint count)
{
    const uint i = get_global_id(0);
    if (i >= count)
        return;

    const uint face = keys[i].y;
    const uint3 f = loadFace(faces, face);
    const float3 a = loadVertex(vertices, f.x);
    const float3 b = loadVertex(vertices, f.y);
    const float3 c = loadVertex(vertices, f.z);
    const AABBGPU box = make_box(fmin(a, fmin(b, c)), fmax(a, fmax(b, c)));

    nodes[count - 1 + i] = (BVHNodeGPU){box, UINT_MAX, UINT_MAX};
    leaf_faces[i] = face;
}

__kernel void lbvh_tile_bounds(__global const BVHNodeGPU* nodes, __global AABBGPU* bounds,
                               uint offset, uint count)
{
    const uint tile = get_global_id(0);
    if (tile >= offset)
        return;

    const uint start = tile * LBVH_TILE_SIZE;
    const uint end = min(start + LBVH_TILE_SIZE, count);
    AABBGPU box = empty_box();
    for (uint i = start; i < end; ++i)
        box = merge_boxes(box, nodes[count - 1 + i].aabb);
    bounds[offset + tile] = box;
}

static inline int prefix_length(__global const uint2* keys, int count, int i, int j)
{
    if (j < 0 || j >= count)
        return -1;
    const uint diff = keys[i].x ^ keys[j].x;
    return diff ? (int)clz(diff) : 32 + (int)clz((uint)(i ^ j));
}

// Range [first, end): scan the two partial tiles and cover the middle
// with aligned tree intervals. Even the root need not scan all its leaves.
static inline AABBGPU interval_bounds(__global const BVHNodeGPU* leaves,
                                       __global const AABBGPU* bounds,
                                       uint offset, uint first, uint end)
{
    uint left = (first + LBVH_TILE_SIZE - 1) / LBVH_TILE_SIZE;
    uint right = end / LBVH_TILE_SIZE;
    const uint head_end = min(end, left * LBVH_TILE_SIZE);
    AABBGPU box = empty_box();
    for (uint i = first; i < head_end; ++i)
        box = merge_boxes(box, leaves[i].aabb);
    for (uint i = max(head_end, right * LBVH_TILE_SIZE); i < end; ++i)
        box = merge_boxes(box, leaves[i].aabb);

    left += offset;
    right += offset;
    while (left < right) {
        if (left & 1u)
            box = merge_boxes(box, bounds[left++]);
        if (right & 1u)
            box = merge_boxes(box, bounds[--right]);
        left /= 2;
        right /= 2;
    }
    return box;
}

__kernel void lbvh_make_internal_nodes(__global const uint2* keys, __global const AABBGPU* bounds,
                                       __global BVHNodeGPU* nodes, uint offset, uint count)
{
    const int i = get_global_id(0);
    const int n = (int)count;
    if (i >= n - 1)
        return;

    // Karras: derive each internal node's range independently from common prefixes.
    const int left_prefix = prefix_length(keys, n, i, i - 1);
    const int right_prefix = prefix_length(keys, n, i, i + 1);
    const int direction = right_prefix > left_prefix ? 1 : -1;
    const int min_prefix = min(left_prefix, right_prefix);
    int limit = 2;
    while (prefix_length(keys, n, i, i + limit * direction) > min_prefix)
        limit *= 2;

    int length = 0;
    for (int step = limit / 2; step > 0; step /= 2) {
        if (prefix_length(keys, n, i, i + (length + step) * direction) > min_prefix)
            length += step;
    }
    const int j = i + length * direction;
    const int first = min(i, j);
    const int last = max(i, j);

    const int common = prefix_length(keys, n, first, last);
    int lo = first;
    int hi = last;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (prefix_length(keys, n, first, mid) > common)
            lo = mid;
        else
            hi = mid;
    }
    const uint split = lo;
    const uint leaf_begin = count - 1;
    const uint left_child = (int)split == first ? leaf_begin + split : split;
    const uint right_child = (int)split + 1 == last ? leaf_begin + split + 1 : split + 1;
    const AABBGPU box = interval_bounds(nodes + leaf_begin, bounds, offset, first, last + 1);
    nodes[i] = (BVHNodeGPU){box, left_child, right_child};
}
