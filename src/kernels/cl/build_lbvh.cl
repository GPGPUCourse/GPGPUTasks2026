#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

#define RADIX_BUCKETS 16
#define RADIX_BLOCK_SIZE (4 * GROUP_SIZE)
#define SCAN_BLOCK_SIZE (4 * GROUP_SIZE)

static inline AABBGPU lbvh_box(float3 lo, float3 hi)
{
    AABBGPU box;
    box.min_x = lo.x;
    box.min_y = lo.y;
    box.min_z = lo.z;
    box.max_x = hi.x;
    box.max_y = hi.y;
    box.max_z = hi.z;
    return box;
}

static inline float3 lbvh_centroid(__global const float* vertices,
                                   __global const uint* faces, uint face)
{
    const uint3 indices = loadFace(faces, face);
    return (loadVertex(vertices, indices.x) + loadVertex(vertices, indices.y)
            + loadVertex(vertices, indices.z)) * (1.0f / 3.0f);
}

static inline void lbvh_reduce_local_bounds(__local float3* lower,
                                            __local float3* upper,
                                            __global AABBGPU* output)
{
    const uint lid = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            lower[lid] = fmin(lower[lid], lower[lid + stride]);
            upper[lid] = fmax(upper[lid], upper[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        output[get_group_id(0)] = lbvh_box(lower[0], upper[0]);
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_centroid_bounds(__global const float* vertices,
                                   __global const uint* faces,
                                   __global AABBGPU* bounds, uint nfaces)
{
    const size_t i = get_global_id(0);
    const uint lid = get_local_id(0);
    __local float3 lower[GROUP_SIZE];
    __local float3 upper[GROUP_SIZE];
    lower[lid] = (float3)(INFINITY);
    upper[lid] = (float3)(-INFINITY);
    if (i < nfaces) {
        lower[lid] = upper[lid] = lbvh_centroid(vertices, faces, (uint)i);
    }
    lbvh_reduce_local_bounds(lower, upper, bounds);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_reduce_bounds(__global const AABBGPU* input,
                                 __global AABBGPU* output, uint count)
{
    const size_t i = get_global_id(0);
    const uint lid = get_local_id(0);
    __local float3 lower[GROUP_SIZE];
    __local float3 upper[GROUP_SIZE];
    lower[lid] = (float3)(INFINITY);
    upper[lid] = (float3)(-INFINITY);
    if (i < count) {
        const AABBGPU box = input[i];
        lower[lid] = (float3)(box.min_x, box.min_y, box.min_z);
        upper[lid] = (float3)(box.max_x, box.max_y, box.max_z);
    }
    lbvh_reduce_local_bounds(lower, upper, output);
}

static inline uint lbvh_expand_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_morton(__global const float* vertices,
                          __global const uint* faces,
                          __global const AABBGPU* scene_bounds,
                          __global uint2* keys, uint nfaces)
{
    const size_t i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    const AABBGPU box = scene_bounds[0];
    const float3 lo = (float3)(box.min_x, box.min_y, box.min_z);
    const float3 hi = (float3)(box.max_x, box.max_y, box.max_z);
    const float3 extent = fmax(hi - lo, (float3)(1e-9f));
    const float3 p = clamp((lbvh_centroid(vertices, faces, (uint)i) - lo) / extent,
                           (float3)(0.0f), (float3)(1.0f));
    const uint3 grid = min(convert_uint3(p * 1024.0f), (uint3)(1023));
    const uint code = (lbvh_expand_bits(grid.x) << 2)
                    | (lbvh_expand_bits(grid.y) << 1) | lbvh_expand_bits(grid.z);
    keys[i] = (uint2)(code, (uint)i);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_count(
    __global const uint2* input,
    __global uint* histogram,
    unsigned int n,
    unsigned int shift)
{
    const uint lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t groups = get_num_groups(0);
    __local uint counts[RADIX_BUCKETS];
    if (lid < RADIX_BUCKETS) {
        counts[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const size_t index = group * RADIX_BLOCK_SIZE + k * GROUP_SIZE + lid;
        if (index < n) {
            atomic_inc(&counts[(input[index].x >> shift) & (RADIX_BUCKETS - 1)]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BUCKETS) {
        histogram[lid * groups + group] = counts[lid];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_scan_reduce(
    __global const uint* input,
    __global uint* block_sums,
    unsigned int n,
    unsigned int input_offset,
    unsigned int output_offset)
{
    input += input_offset;
    block_sums += output_offset;
    const uint lid = get_local_id(0);
    const size_t base = get_group_id(0) * SCAN_BLOCK_SIZE + 4 * lid;
    uint4 values = (uint4)(0);
    if (base + 3 < n) {
        values = vload4(0, input + base);
    }
    else {
        for (uint i = 0; i < 4; ++i) {
            if (base + i < n) {
                values[i] = input[base + i];
            }
        }
    }
    __local uint sums[GROUP_SIZE];
    sums[lid] = values.x + values.y + values.z + values.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sums[lid] += sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        block_sums[get_group_id(0)] = sums[0];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_scan(
    __global const uint* input,
    __global const uint* block_prefixes,
    __global uint* output,
    unsigned int n,
    unsigned int data_offset,
    unsigned int prefixes_offset)
{
    input += data_offset;
    output += data_offset;
    const uint lid = get_local_id(0);
    const size_t block = get_group_id(0);
    const size_t base = block * SCAN_BLOCK_SIZE + 4 * lid;
    uint4 values = (uint4)(0);
    if (base + 3 < n) {
        values = vload4(0, input + base);
    }
    else {
        for (uint i = 0; i < 4; ++i) {
            if (base + i < n) {
                values[i] = input[base + i];
            }
        }
    }
    __local uint sums[GROUP_SIZE];
    sums[lid] = values.x + values.y + values.z + values.w;

    values = (uint4)(0, values.x, values.x + values.y, values.x + values.y + values.z);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        const uint index = (lid + 1) * 2 * stride - 1;
        if (index < GROUP_SIZE) {
            sums[index] += sums[index - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        sums[GROUP_SIZE - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        const uint index = (lid + 1) * 2 * stride - 1;
        if (index < GROUP_SIZE) {
            const uint left = sums[index - stride];
            sums[index - stride] = sums[index];
            sums[index] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint offset = sums[lid] + (get_num_groups(0) == 1
        ? 0 : block_prefixes[prefixes_offset + block]);
    values += (uint4)(offset);
    if (base + 3 < n) {
        vstore4(values, 0, output + base);
    }
    else {
        for (uint i = 0; i < 4; ++i) {
            if (base + i < n) {
                output[base + i] = values[i];
            }
        }
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_scatter(
    __global const uint2* input,
    __global const uint* prefixes,
    __global uint2* output,
    unsigned int n,
    unsigned int shift)
{
    const uint lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t groups = get_num_groups(0);
    const size_t base = group * RADIX_BLOCK_SIZE;

    const uint chunks = RADIX_BLOCK_SIZE / 32;
    __local uint masks[RADIX_BUCKETS * (RADIX_BLOCK_SIZE / 32)];
    __local uint chunk_prefixes[RADIX_BUCKETS * (RADIX_BLOCK_SIZE / 32)];
    __local uint starts[RADIX_BUCKETS];
    __local uint2 ordered[RADIX_BLOCK_SIZE];
    uint2 values[RADIX_BLOCK_SIZE / GROUP_SIZE];

    for (uint i = lid; i < RADIX_BUCKETS * chunks; i += GROUP_SIZE) {
        masks[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const uint position = k * GROUP_SIZE + lid;
        if (base + position < n) {
            values[k] = input[base + position];
            const uint digit = (values[k].x >> shift) & (RADIX_BUCKETS - 1);
            atomic_or(&masks[digit * chunks + position / 32], 1u << (position % 32));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BUCKETS) {
        uint sum = 0;
        for (uint chunk = 0; chunk < chunks; ++chunk) {
            chunk_prefixes[lid * chunks + chunk] = sum;
            sum += popcount(masks[lid * chunks + chunk]);
        }
        starts[lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        uint sum = 0;
        for (uint digit = 0; digit < RADIX_BUCKETS; ++digit) {
            const uint count = starts[digit];
            starts[digit] = sum;
            sum += count;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const uint position = k * GROUP_SIZE + lid;
        if (base + position < n) {
            const uint digit = (values[k].x >> shift) & (RADIX_BUCKETS - 1);
            const uint chunk = digit * chunks + position / 32;
            const uint earlier = (1u << (position % 32)) - 1;
            const uint rank = chunk_prefixes[chunk] + popcount(masks[chunk] & earlier);
            ordered[starts[digit] + rank] = values[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const uint position = k * GROUP_SIZE + lid;
        if (base + position < n) {
            const uint2 value = ordered[position];
            const uint digit = (value.x >> shift) & (RADIX_BUCKETS - 1);
            const size_t destination = prefixes[digit * groups + group] + position - starts[digit];
            output[destination] = value;
        }
    }
}

static inline int lbvh_common_prefix(__global const uint2* keys, uint n,
                                     uint i, long j)
{
    if (j < 0 || j >= (long)n) {
        return -1;
    }
    const uint different = keys[i].x ^ keys[j].x;
    return different != 0 ? (int)clz(different) : 32 + (int)clz(i ^ (uint)j);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_topology(__global const float* vertices,
                            __global const uint* faces,
                            __global const uint2* keys,
                            __global BVHNodeGPU* nodes,
                            __global uint* leaf_indices,
                            __global uint* prefix_lengths, uint nfaces)
{
    const size_t gid = get_global_id(0);
    if (gid >= nfaces) {
        return;
    }
    const uint i = (uint)gid;
    const uint face = keys[i].y;
    const uint3 indices = loadFace(faces, face);
    const float3 a = loadVertex(vertices, indices.x);
    const float3 b = loadVertex(vertices, indices.y);
    const float3 c = loadVertex(vertices, indices.z);
    const uint leaf = nfaces - 1 + i;
    nodes[leaf].aabb = lbvh_box(fmin(a, fmin(b, c)), fmax(a, fmax(b, c)));
    nodes[leaf].leftChildIndex = 0xffffffffu;
    nodes[leaf].rightChildIndex = 0xffffffffu;
    leaf_indices[i] = face;
    if (i == nfaces - 1) {
        return;
    }

    const int left_prefix = lbvh_common_prefix(keys, nfaces, i, (long)i - 1);
    const int right_prefix = lbvh_common_prefix(keys, nfaces, i, (long)i + 1);
    const int direction = right_prefix > left_prefix ? 1 : -1;
    const int minimum = lbvh_common_prefix(keys, nfaces, i, (long)i - direction);
    uint limit = 2;
    while (lbvh_common_prefix(keys, nfaces, i, (long)i + (long)limit * direction) > minimum) {
        limit <<= 1;
    }
    uint length = 0;
    for (uint step = limit >> 1; step > 0; step >>= 1) {
        if (lbvh_common_prefix(keys, nfaces, i, (long)i + (long)(length + step) * direction) > minimum) {
            length += step;
        }
    }
    const uint other = (uint)((long)i + (long)length * direction);
    const uint first = min(i, other);
    const uint last = max(i, other);
    const int prefix = lbvh_common_prefix(keys, nfaces, first, last);

    uint split = first;
    uint step = last - first;
    do {
        step = (step + 1) >> 1;
        const uint candidate = split + step;
        if (candidate < last && lbvh_common_prefix(keys, nfaces, first, candidate) > prefix) {
            split = candidate;
        }
    } while (step > 1);
    nodes[i].leftChildIndex = split == first ? nfaces - 1 + split : split;
    nodes[i].rightChildIndex = split + 1 == last ? nfaces - 1 + split + 1 : split + 1;
    prefix_lengths[i] = (uint)prefix;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_bounds(__global BVHNodeGPU* nodes,
                          __global const uint* prefix_lengths,
                          uint nfaces, uint level)
{
    const size_t i = get_global_id(0);
    if (i >= nfaces - 1 || prefix_lengths[i] != level) {
        return;
    }
    const AABBGPU left = nodes[nodes[i].leftChildIndex].aabb;
    const AABBGPU right = nodes[nodes[i].rightChildIndex].aabb;
    nodes[i].aabb = lbvh_box(
        fmin((float3)(left.min_x, left.min_y, left.min_z),
             (float3)(right.min_x, right.min_y, right.min_z)),
        fmax((float3)(left.max_x, left.max_y, left.max_z),
             (float3)(right.max_x, right.max_y, right.max_z)));
}
