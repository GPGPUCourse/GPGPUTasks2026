#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline uint expand_bits(uint value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

static inline uint morton3d(float x, float y, float z)
{
    const uint xx = expand_bits((uint)(clamp(x, 0.0f, 1.0f) * 1023.0f));
    const uint yy = expand_bits((uint)(clamp(y, 0.0f, 1.0f) * 1023.0f));
    const uint zz = expand_bits((uint)(clamp(z, 0.0f, 1.0f) * 1023.0f));
    return xx * 4u + yy * 2u + zz;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_compute_morton(
    __global const float* vertices,
    __global const uint* faces,
    __global uint* morton_codes,
    __global uint* triangle_indices,
    uint nfaces,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z)
{
    const uint i = get_global_id(0);
    if (i >= nfaces) return;

    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    const float3 center = (a + b + c) * (1.0f / 3.0f);
    const float dx = fmax(max_x - min_x, 1e-9f);
    const float dy = fmax(max_y - min_y, 1e-9f);
    const float dz = fmax(max_z - min_z, 1e-9f);

    morton_codes[i] = morton3d(
        (center.x - min_x) / dx,
        (center.y - min_y) / dy,
        (center.z - min_z) / dz);
    triangle_indices[i] = i;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_radix_local_counting(
    __global const uint* codes,
    __global uint* local_zero_prefixes,
    __global uint* group_zero_counts,
    uint n,
    uint bit)
{
    const uint global_index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint prefixes[GROUP_SIZE];
    prefixes[local_index] = global_index < n
        ? 1u - ((codes[global_index] >> bit) & 1u)
        : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1u; offset < GROUP_SIZE; offset *= 2u) {
        const uint value = local_index >= offset ? prefixes[local_index - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        prefixes[local_index] += value;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_index < n) local_zero_prefixes[global_index] = prefixes[local_index];
    if (local_index == GROUP_SIZE - 1u) group_zero_counts[get_group_id(0)] = prefixes[local_index];
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_scan_first(
    __global const uint* input,
    __global uint* output,
    uint n)
{
    const uint i = get_global_id(0);
    if (i < n) output[i] = input[i] + (i > 0u ? input[i - 1u] : 0u);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_scan_step(
    __global const uint* input,
    __global uint* output,
    uint n,
    uint offset)
{
    const uint i = get_global_id(0);
    if (i < n) output[i] = input[i] + (i >= offset ? input[i - offset] : 0u);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_radix_scatter(
    __global const uint* input_codes,
    __global const uint* input_indices,
    __global const uint* local_zero_prefixes,
    __global const uint* group_zero_prefixes,
    __global uint* output_codes,
    __global uint* output_indices,
    uint n,
    uint bit,
    uint groups_count)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint group = get_group_id(0);
    const uint zeros_before_group = group > 0u ? group_zero_prefixes[group - 1u] : 0u;
    const uint local_zero_prefix = local_zero_prefixes[i];
    const uint total_zeros = group_zero_prefixes[groups_count - 1u];
    const uint is_one = (input_codes[i] >> bit) & 1u;
    const uint destination = is_one
        ? total_zeros + i - zeros_before_group - local_zero_prefix
        : zeros_before_group + local_zero_prefix - 1u;
    output_codes[destination] = input_codes[i];
    output_indices[destination] = input_indices[i];
}

static inline int common_prefix(__global const uint* codes, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;
    const uint a = codes[i];
    const uint b = codes[j];
    return a == b ? 32 + clz((uint)(i ^ j)) : clz(a ^ b);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_build_topology(
    __global const uint* sorted_codes,
    __global BVHNodeGPU* nodes,
    __global uint* parents,
    uint n)
{
    const int i = (int)get_global_id(0);
    if (i >= (int)n - 1) return;

    const int left_prefix = common_prefix(sorted_codes, (int)n, i, i - 1);
    const int right_prefix = common_prefix(sorted_codes, (int)n, i, i + 1);
    const int direction = right_prefix > left_prefix ? 1 : -1;
    const int minimum_prefix = common_prefix(sorted_codes, (int)n, i, i - direction);
    int maximum_length = 2;
    while (common_prefix(sorted_codes, (int)n, i, i + maximum_length * direction) > minimum_prefix) {
        maximum_length *= 2;
    }
    int length = 0;
    for (int step = maximum_length / 2; step > 0; step /= 2) {
        if (common_prefix(sorted_codes, (int)n, i, i + (length + step) * direction) > minimum_prefix) {
            length += step;
        }
    }
    const int other = i + length * direction;
    const int first = min(i, other);
    const int last = max(i, other);
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
    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    parents[left] = (uint)i;
    parents[right] = (uint)i;
    if (i == 0) parents[0] = 0xFFFFFFFFu;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_init_leaves(
    __global const float* vertices,
    __global const uint* faces,
    __global const uint* sorted_triangle_indices,
    __global BVHNodeGPU* nodes,
    uint n)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const uint face_id = sorted_triangle_indices[i];
    const uint3 face = loadFace(faces, face_id);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    const uint node_index = n - 1u + i;
    nodes[node_index].aabb.min_x = fmin(a.x, fmin(b.x, c.x));
    nodes[node_index].aabb.min_y = fmin(a.y, fmin(b.y, c.y));
    nodes[node_index].aabb.min_z = fmin(a.z, fmin(b.z, c.z));
    nodes[node_index].aabb.max_x = fmax(a.x, fmax(b.x, c.x));
    nodes[node_index].aabb.max_y = fmax(a.y, fmax(b.y, c.y));
    nodes[node_index].aabb.max_z = fmax(a.z, fmax(b.z, c.z));
    nodes[node_index].leftChildIndex = 0xFFFFFFFFu;
    nodes[node_index].rightChildIndex = 0xFFFFFFFFu;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_clear_counters(__global uint* counters, uint n)
{
    const uint i = get_global_id(0);
    if (i < n) counters[i] = 0u;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_build_aabbs(
    __global BVHNodeGPU* nodes,
    __global const uint* parents,
    volatile __global uint* counters,
    uint n)
{
    const uint leaf = get_global_id(0);
    if (leaf >= n || n <= 1u) return;
    uint current = n - 1u + leaf;

    while (parents[current] != 0xFFFFFFFFu) {
        const uint parent = parents[current];
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        if (atomic_inc(&counters[parent]) == 0u) return;

        const BVHNodeGPU left = nodes[nodes[parent].leftChildIndex];
        const BVHNodeGPU right = nodes[nodes[parent].rightChildIndex];
        nodes[parent].aabb.min_x = fmin(left.aabb.min_x, right.aabb.min_x);
        nodes[parent].aabb.min_y = fmin(left.aabb.min_y, right.aabb.min_y);
        nodes[parent].aabb.min_z = fmin(left.aabb.min_z, right.aabb.min_z);
        nodes[parent].aabb.max_x = fmax(left.aabb.max_x, right.aabb.max_x);
        nodes[parent].aabb.max_y = fmax(left.aabb.max_y, right.aabb.max_y);
        nodes[parent].aabb.max_z = fmax(left.aabb.max_z, right.aabb.max_z);
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        current = parent;
    }
}
