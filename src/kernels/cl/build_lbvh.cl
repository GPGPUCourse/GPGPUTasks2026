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

__kernel void triangle_bounds(
    __global const float* vertices,
    __global const uint* faces,
    __global AABBGPU* boxes,
    __global float* centroids,
    __global float* bounds,
    uint faceCount)
{
    uint faceIndex = get_global_id(0);
    if (faceIndex >= faceCount)
        return;

    uint3 face = loadFace(faces, faceIndex);
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
    boxes[faceIndex] = box;

    float3 center = (a + b + c) * (1.0f / 3.0f);
    centroids[3 * faceIndex] = center.x;
    centroids[3 * faceIndex + 1] = center.y;
    centroids[3 * faceIndex + 2] = center.z;

    bounds[6 * faceIndex] = center.x;
    bounds[6 * faceIndex + 1] = center.y;
    bounds[6 * faceIndex + 2] = center.z;
    bounds[6 * faceIndex + 3] = center.x;
    bounds[6 * faceIndex + 4] = center.y;
    bounds[6 * faceIndex + 5] = center.z;
}

__kernel void reduce_bounds(
    __global const float* input,
    __global float* output,
    uint inputCount)
{
    uint groupIndex = get_global_id(0);
    uint first = groupIndex * 256;
    if (first >= inputCount)
        return;

    float minimum[3] = {INFINITY, INFINITY, INFINITY};
    float maximum[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (uint i = first; i < min(first + 256, inputCount); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = fmin(minimum[axis], input[6 * i + axis]);
            maximum[axis] = fmax(maximum[axis], input[6 * i + 3 + axis]);
        }
    }

    for (int axis = 0; axis < 3; ++axis) {
        output[6 * groupIndex + axis] = minimum[axis];
        output[6 * groupIndex + 3 + axis] = maximum[axis];
    }
}

__kernel void morton_codes(
    __global const float* centroids,
    __global const float* bounds,
    __global uint* codes,
    __global uint* indices,
    uint faceCount,
    uint paddedCount)
{
    uint index = get_global_id(0);
    if (index >= paddedCount)
        return;

    indices[index] = index;
    if (index >= faceCount) {
        codes[index] = UINT_MAX;
        return;
    }

    uint coordinates[3];
    for (int axis = 0; axis < 3; ++axis) {
        float extent = fmax(bounds[3 + axis] - bounds[axis], 1e-9f);
        float normalized = clamp((centroids[3 * index + axis] - bounds[axis]) / extent, 0.0f, 1.0f);
        coordinates[axis] = min((uint)(normalized * 1024.0f), 1023u);
    }

    codes[index] = (expand_bits(coordinates[0]) << 2) | (expand_bits(coordinates[1]) << 1) | expand_bits(coordinates[2]);
}

__kernel void sort_codes(
    __global uint* codes,
    __global uint* indices,
    uint paddedCount,
    uint width,
    uint stride)
{
    uint left = get_global_id(0);
    if (left >= paddedCount)
        return;

    uint right = left ^ stride;
    if (right < left)
        return;

    bool leftGreater = codes[left] > codes[right] || (codes[left] == codes[right] && indices[left] > indices[right]);
    bool ascending = (left & width) == 0;
    if (leftGreater != ascending)
        return;

    uint code = codes[left];
    codes[left] = codes[right];
    codes[right] = code;

    uint index = indices[left];
    indices[left] = indices[right];
    indices[right] = index;
}

static inline int common_prefix(
    __global const uint* codes,
    int count,
    int first,
    int second)
{
    if (second < 0 || second >= count)
        return -1;

    uint firstCode = codes[first];
    uint secondCode = codes[second];
    if (firstCode == secondCode)
        return 32 + clz((uint)(first ^ second));

    return clz(firstCode ^ secondCode);
}

__kernel void init_leaves(
    __global const AABBGPU* boxes,
    __global const uint* indices,
    __global BVHNodeGPU* nodes,
    __global uint* leafIndices,
    uint faceCount)
{
    uint leafIndex = get_global_id(0);
    if (leafIndex >= faceCount)
        return;

    uint faceIndex = indices[leafIndex];
    uint nodeIndex = faceCount - 1 + leafIndex;

    nodes[nodeIndex].aabb = boxes[faceIndex];
    nodes[nodeIndex].leftChildIndex = UINT_MAX;
    nodes[nodeIndex].rightChildIndex = UINT_MAX;
    leafIndices[leafIndex] = faceIndex;
}

__kernel void link_nodes(
    __global const uint* codes,
    __global BVHNodeGPU* nodes,
    __global uint* parents,
    uint faceCount)
{
    int nodeIndex = get_global_id(0);
    if (nodeIndex >= (int)faceCount - 1)
        return;

    int leftPrefix = common_prefix(codes, faceCount, nodeIndex, nodeIndex - 1);
    int rightPrefix = common_prefix(codes, faceCount, nodeIndex, nodeIndex + 1);
    int direction = rightPrefix > leftPrefix ? 1 : -1;
    int minimumPrefix = common_prefix(codes, faceCount, nodeIndex, nodeIndex - direction);

    int maximumLength = 2;
    while (common_prefix(codes, faceCount, nodeIndex, nodeIndex + maximumLength * direction) > minimumPrefix)
        maximumLength *= 2;

    int length = 0;
    for (int step = maximumLength / 2; step > 0; step /= 2) {
        if (common_prefix(codes, faceCount, nodeIndex, nodeIndex + (length + step) * direction) > minimumPrefix)
            length += step;
    }

    int first = min(nodeIndex, nodeIndex + length * direction);
    int last = max(nodeIndex, nodeIndex + length * direction);
    int rangePrefix = common_prefix(codes, faceCount, first, last);

    int split = first;
    int step = last - first;
    do {
        step = (step + 1) / 2;
        int next = split + step;
        if (next < last && common_prefix(codes, faceCount, first, next) > rangePrefix)
            split = next;
    } while (step > 1);

    uint leftChild = split == first ? faceCount - 1 + split : split;
    uint rightChild = split + 1 == last ? faceCount + split : split + 1;

    nodes[nodeIndex].leftChildIndex = leftChild;
    nodes[nodeIndex].rightChildIndex = rightChild;
    parents[leftChild] = nodeIndex;
    parents[rightChild] = nodeIndex;
}

__kernel void propagate_bounds(
    __global BVHNodeGPU* nodes,
    __global const uint* parents,
    __global uint* arrivals,
    uint faceCount)
{
    uint leafIndex = get_global_id(0);
    if (leafIndex >= faceCount || faceCount == 1)
        return;

    uint child = faceCount - 1 + leafIndex;
    while (child != 0) {
        uint parent = parents[child];

        mem_fence(CLK_GLOBAL_MEM_FENCE);
        if (atomic_inc(&arrivals[parent]) == 0)
            return;

        mem_fence(CLK_GLOBAL_MEM_FENCE);
        AABBGPU left = nodes[nodes[parent].leftChildIndex].aabb;
        AABBGPU right = nodes[nodes[parent].rightChildIndex].aabb;

        AABBGPU box;
        box.min_x = fmin(left.min_x, right.min_x);
        box.min_y = fmin(left.min_y, right.min_y);
        box.min_z = fmin(left.min_z, right.min_z);
        box.max_x = fmax(left.max_x, right.max_x);
        box.max_y = fmax(left.max_y, right.max_y);
        box.max_z = fmax(left.max_z, right.max_z);

        nodes[parent].aabb = box;
        child = parent;
    }
}
