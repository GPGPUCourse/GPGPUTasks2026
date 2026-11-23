#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

#define INVALID_INDEX 0xFFFFFFFFu
#define RADIX_SIZE 16
#define epsilon 1.0e-9f

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void calculating_triangle_aabb(
    __global const float* vertices,
    __global const uint* faces,
    uint nfaces,
    __global AABBGPU* trianglesAABB,
    __global float* centroids)
{
    uint gid = get_global_id(0);
    if (gid >= nfaces) {
        return;
    }
    uint3 triangle = loadFace(faces, gid);
    float3 v0 = loadVertex(vertices, triangle.x);
    float3 v1 = loadVertex(vertices, triangle.y);
    float3 v2 = loadVertex(vertices, triangle.z);

    float tmp = min(v0.x, v1.x);
    float min_x = min(tmp, v2.x);
    tmp = min(v0.y, v1.y);
    float min_y = min(tmp, v2.y);
    tmp = min(v0.z, v1.z);
    float min_z = min(tmp, v2.z);

    tmp = max(v0.x, v1.x);
    float max_x = max(tmp, v2.x);
    tmp = max(v0.y, v1.y);
    float max_y = max(tmp, v2.y);
    tmp = max(v0.z, v1.z);
    float max_z = max(tmp, v2.z);
    
    trianglesAABB[gid].min_x = min_x;
    trianglesAABB[gid].min_y = min_y;
    trianglesAABB[gid].min_z = min_z;

    trianglesAABB[gid].max_x = max_x;
    trianglesAABB[gid].max_y = max_y;
    trianglesAABB[gid].max_z = max_z;

    float centroidX = (v0.x + v1.x + v2.x)/3.0f;
    float centroidY = (v0.y + v1.y + v2.y)/3.0f;
    float centroidZ = (v0.z + v1.z + v2.z)/3.0f;

    centroids[3 * gid] = centroidX;
    centroids[3 * gid + 1] = centroidY;
    centroids[3 * gid + 2] = centroidZ;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void initialize_triangle_indices(
    uint nfaces,
    __global uint* indices)
{
    uint gid = get_global_id(0);

    if (gid >= nfaces)
        return;

    indices[gid] = gid;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void local_reduction(
    uint nfaces,
    __global const float* centroids,
    __global float* buffer)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);
    __local float3 localMin[GROUP_SIZE];
    __local float3 localMax[GROUP_SIZE];

    if (gid < nfaces) {
        localMin[lid] = (float3)(centroids[gid*3], centroids[gid*3 + 1], centroids[gid*3 + 2]);
        localMax[lid] = (float3)(centroids[gid*3], centroids[gid*3 + 1], centroids[gid*3 + 2]);
    } else {
        localMin[lid] = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
        localMax[lid] = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint stride = GROUP_SIZE/2;
    while (stride > 0) {
        if (lid < stride) {
            localMin[lid] = fmin(localMin[lid], localMin[lid + stride]);
            localMax[lid] = fmax(localMax[lid], localMax[lid + stride]);
        }
        stride /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        buffer[6*group_id] = localMin[0].x;
        buffer[6*group_id + 1] = localMin[0].y;
        buffer[6*group_id + 2] = localMin[0].z;

        buffer[6*group_id + 3] = localMax[0].x;
        buffer[6*group_id + 4] = localMax[0].y;
        buffer[6*group_id + 5] = localMax[0].z;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void global_reduction(
    __global const float* coordinates_tmp,
    uint num_groups,
    __global float* out)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);

    __local float3 localMin[GROUP_SIZE];
    __local float3 localMax[GROUP_SIZE];

    if (gid < num_groups) {
        localMin[lid] = (float3)(coordinates_tmp[gid*6], coordinates_tmp[gid*6 + 1], coordinates_tmp[gid*6 + 2]);
        localMax[lid] = (float3)(coordinates_tmp[gid*6 + 3], coordinates_tmp[gid*6 + 4], coordinates_tmp[gid*6 + 5]);
    } else {
        localMin[lid] = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
        localMax[lid] = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint stride = GROUP_SIZE/2;
    while (stride > 0) {
        if (lid < stride) {
            localMin[lid] = fmin(localMin[lid], localMin[lid + stride]);
            localMax[lid] = fmax(localMax[lid], localMax[lid + stride]);
        }
        stride /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[6*group_id] = localMin[0].x;
        out[6*group_id + 1] = localMin[0].y;
        out[6*group_id + 2] = localMin[0].z;

        out[6*group_id + 3] = localMax[0].x;
        out[6*group_id + 4] = localMax[0].y;
        out[6*group_id + 5] = localMax[0].z;
    }
}

unsigned int expandBits(unsigned int v)
{
    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void morton_code(
    __global const float* centroids,
    __global const float* coordinates,
    uint nfaces,
    __global uint* morton_codes)
{
    uint gid = get_global_id(0);
    if (gid >= nfaces) {
        return;
    }
    float centroidX = centroids[gid * 3];
    float centroidY = centroids[gid * 3 + 1];
    float centroidZ = centroids[gid * 3 + 2];

    float minX = coordinates[0];
    float minY = coordinates[2];
    float minZ = coordinates[4];

    float dx = coordinates[1] - minX;
    float dy = coordinates[3] - minY;
    float dz = coordinates[5] - minZ;

    dx = fmax(dx, epsilon);
    dy = fmax(dy, epsilon);
    dz = fmax(dz, epsilon);

    float normalizedX = (centroidX - minX)/dx;
    float normalizedY = (centroidY - minY)/dy;
    float normalizedZ = (centroidZ - minZ)/dz;

    uint ix = (uint)clamp((int)(clamp(normalizedX, 0.0f, 1.0f) * 1024.0f), 0, 1023);
    uint iy = (uint)clamp((int)(clamp(normalizedY, 0.0f, 1.0f) * 1024.0f), 0, 1023);
    uint iz = (uint)clamp((int)(clamp(normalizedZ, 0.0f, 1.0f) * 1024.0f), 0, 1023);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);

    uint morton_code = (xx << 2) | (yy << 1) | (zz);
    morton_codes[gid] = morton_code;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global uint* local_rank,
    __global uint* histogram,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    const uint valid = gid < n;
    const uint value = valid ? values[gid] : 0u;
    const uint digit = (value >> bit) & 15u;

    __local uint scan[GROUP_SIZE];

    for (uint d = 0; d < RADIX_SIZE; ++d) {
        const uint matches = valid && digit == d;

        scan[lid] = matches;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
            const uint add =
                lid >= offset ? scan[lid - offset] : 0u;

            barrier(CLK_LOCAL_MEM_FENCE);
            scan[lid] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (valid && digit == d) {
            local_rank[gid] = scan[lid] - 1u;
        }

        if (lid == GROUP_SIZE - 1) {
            histogram[d * num_groups + group] = scan[lid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* histogram,
    __global uint* totals,
    unsigned int n)
{
    const uint digit = get_global_id(0);
    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    if (digit >= RADIX_SIZE)
        return;

    uint total = 0u;

    for (uint group = 0; group < num_groups; ++group) {
        total += histogram[digit * num_groups + group];
    }

    totals[digit] = total;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global uint* offsets,
    __global const uint* totals,
    unsigned int n)
{
    if (get_global_id(0) != 0)
        return;

    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    uint global_base = 0u;

    for (uint digit = 0; digit < RADIX_SIZE; ++digit) {
        uint block_prefix = 0u;

        for (uint group = 0; group < num_groups; ++group) {
            const uint index = digit * num_groups + group;
            const uint count = offsets[index];

            offsets[index] = global_base + block_prefix;
            block_prefix += count;
        }

        global_base += totals[digit];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* morton,
    __global const uint* idx_triangle,
    __global const uint* local_rank,
    __global const uint* offsets,
    __global uint* output_morton,
    __global uint* output_triangles,
    unsigned int n,
    unsigned int bit)
{
    const uint gid = get_global_id(0);

    if (gid >= n)
        return;

    const uint num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    const uint group = gid / GROUP_SIZE;

    const uint code = morton[gid];
    const uint idx = idx_triangle[gid];
    const uint digit = (code >> bit) & 15u;
    const uint rank = local_rank[gid];

    const uint dst = offsets[digit * num_groups + group] + rank;

    output_morton[dst] = code;
    output_triangles[dst] = idx;
}

static inline int common_prefix(__global const uint* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    uint ci = codes[i];
    uint cj = codes[j];

    if (ci == cj) {
        uint di = (uint)(i);
        uint dj = (uint)(j);
        uint diff = di ^ dj;
        return 32 + clz(diff);
    } else {
        uint diff = ci ^ cj;
        return clz(diff);
    }
}

// Determine range [first, last] of primitives covered by internal node i
static inline void determine_range(__global const uint* codes, int N, int i, int* outFirst, int* outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    *outFirst = min(i, j);
    *outLast  = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
static inline int find_split(__global const uint* codes, int N,
                             int first, int last)
{
    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void build_lbvh_topology(
    __global const uint* codes,
    const uint n,
    __global BVHNodeGPU* nodes)
{
    uint gid = get_global_id(0);
    if (gid >= n - 1) {
        return;
    }

    int first, last;
    determine_range(codes, (int)n, (int)gid, &first, &last);
    int split = find_split(codes, (int)n, first, last);

    uint leftIndex = split;
    uint rightIndex = split + 1;
    if (split == first) {
        leftIndex = (n - 1) + split; 
    }
    if (split + 1 == last) {
        rightIndex = (n - 1) + split + 1;
    }

    nodes[gid].leftChildIndex = leftIndex;
    nodes[gid].rightChildIndex = rightIndex;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void build_parent_indices(
    __global const BVHNodeGPU* nodes,
    uint n,
    __global uint* parent)
{
    uint gid = get_global_id(0);
    if (gid >= n - 1) return;

    uint left = nodes[gid].leftChildIndex;
    uint right = nodes[gid].rightChildIndex;

    parent[left] = gid;
    parent[right] = gid;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void initialize_leaf_nodes(
    uint nfaces,
    __global const uint* sorted_triangle_indices,
    __global const AABBGPU* triangle_aabbs,
    __global BVHNodeGPU* nodes)
{
    uint gid = get_global_id(0);

    if (gid >= nfaces)
        return;

    uint triangle_index = sorted_triangle_indices[gid];
    uint leaf_index = nfaces - 1 + gid;

    nodes[leaf_index].aabb = triangle_aabbs[triangle_index];
    nodes[leaf_index].leftChildIndex = INVALID_INDEX;
    nodes[leaf_index].rightChildIndex = INVALID_INDEX;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void compute_internal_aabbs(
    __global const uint* codes,
    uint nfaces,
    __global BVHNodeGPU* nodes)
{
    uint gid = get_global_id(0);

    if (gid >= nfaces - 1)
        return;

    int first;
    int last;

    determine_range(
        codes,
        (int)nfaces,
        (int)gid,
        &first,
        &last
    );

    AABBGPU result;

    result.min_x = FLT_MAX;
    result.min_y = FLT_MAX;
    result.min_z = FLT_MAX;

    result.max_x = -FLT_MAX;
    result.max_y = -FLT_MAX;
    result.max_z = -FLT_MAX;

    for (int i = first; i <= last; ++i)
    {
        uint leaf_index = nfaces - 1 + (uint)i;
        AABBGPU leaf = nodes[leaf_index].aabb;

        result.min_x = fmin(result.min_x, leaf.min_x);
        result.min_y = fmin(result.min_y, leaf.min_y);
        result.min_z = fmin(result.min_z, leaf.min_z);

        result.max_x = fmax(result.max_x, leaf.max_x);
        result.max_y = fmax(result.max_y, leaf.max_y);
        result.max_z = fmax(result.max_z, leaf.max_z);
    }

    nodes[gid].aabb = result;
}