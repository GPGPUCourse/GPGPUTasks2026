#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "geometry_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void triangle_bounds(
    __global const float* vertices,
    __global const uint* faces,
    __global AABBGPU* triangle_bounding_boxes,
    __global float* triangle_centroids,
    uint face_count)
{
    const uint face_index = get_global_id(0);

    if (face_index >= face_count) {
        return;
    }

    const uint3 vertex_indices = loadFace(faces, face_index);
    const float3 first_vertex = loadVertex(vertices, vertex_indices.x);
    const float3 second_vertex = loadVertex(vertices, vertex_indices.y);
    const float3 third_vertex = loadVertex(vertices, vertex_indices.z);

    const float3 minimum_corner = fmin(
        fmin(first_vertex, second_vertex),
        third_vertex);

    const float3 maximum_corner = fmax(
        fmax(first_vertex, second_vertex),
        third_vertex);

    const float3 centroid = (first_vertex + second_vertex + third_vertex) * (1.0f / 3.0f);

    const AABBGPU bounding_box = {
        minimum_corner.x,
        minimum_corner.y,
        minimum_corner.z,
        maximum_corner.x,
        maximum_corner.y,
        maximum_corner.z,
    };

    triangle_bounding_boxes[face_index] = bounding_box;
    vstore3(centroid, face_index, triangle_centroids);
}

static inline void reduce_group_bounds(
    __local AABBGPU* group_bounds,
    __global AABBGPU* output_bounds)
{
    const uint local_index = get_local_id(0);

    for (uint stride = GROUP_SIZE / 2; stride > 0; stride /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_index < stride) {
            const AABBGPU first_bounds = group_bounds[local_index];
            const AABBGPU second_bounds = group_bounds[local_index + stride];

            const AABBGPU combined_bounds = {
                fmin(first_bounds.min_x, second_bounds.min_x),
                fmin(first_bounds.min_y, second_bounds.min_y),
                fmin(first_bounds.min_z, second_bounds.min_z),
                fmax(first_bounds.max_x, second_bounds.max_x),
                fmax(first_bounds.max_y, second_bounds.max_y),
                fmax(first_bounds.max_z, second_bounds.max_z),
            };

            group_bounds[local_index] = combined_bounds;
        }
    }

    if (local_index == 0) {
        output_bounds[get_group_id(0)] = group_bounds[0];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void reduce_centroid_bounds(
    __global const float* triangle_centroids,
    __global AABBGPU* output_bounds,
    uint centroid_count)
{
    const uint centroid_index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local AABBGPU group_bounds[GROUP_SIZE];

    AABBGPU bounds = {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};

    if (centroid_index < centroid_count) {
        const float3 centroid = vload3(centroid_index, triangle_centroids);

        bounds = (AABBGPU){
            centroid.x,
            centroid.y,
            centroid.z,
            centroid.x,
            centroid.y,
            centroid.z,
        };
    }

    group_bounds[local_index] = bounds;
    reduce_group_bounds(group_bounds, output_bounds);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void reduce_bounds(
    __global const AABBGPU* input_bounds,
    __global AABBGPU* output_bounds,
    uint bounds_count)
{
    const uint bounds_index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local AABBGPU group_bounds[GROUP_SIZE];

    const AABBGPU empty_bounds = {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
    group_bounds[local_index] = bounds_index < bounds_count
        ? input_bounds[bounds_index]
        : empty_bounds;

    reduce_group_bounds(group_bounds, output_bounds);
}

static inline uint expand_morton_bits(uint coordinate)
{
    coordinate = (coordinate * 0x00010001u) & 0xFF0000FFu;
    coordinate = (coordinate * 0x00000101u) & 0x0F00F00Fu;
    coordinate = (coordinate * 0x00000011u) & 0xC30C30C3u;
    coordinate = (coordinate * 0x00000005u) & 0x49249249u;

    return coordinate;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void morton_codes(
    __global const float* triangle_centroids,
    __global const AABBGPU* centroid_bounds,
    __global uint2* morton_pairs,
    uint face_count)
{
    const uint face_index = get_global_id(0);

    if (face_index >= face_count) {
        return;
    }

    const AABBGPU bounds = centroid_bounds[0];

    const float3 minimum_corner = (float3)(bounds.min_x, bounds.min_y, bounds.min_z);
    const float3 maximum_corner = (float3)(bounds.max_x, bounds.max_y, bounds.max_z);
    
    const float3 extent = fmax(maximum_corner - minimum_corner, (float3)(1e-9f));
    const float3 centroid = vload3(face_index, triangle_centroids);

    const float3 normalized_centroid = clamp(
        (centroid - minimum_corner) / extent,
        0.0f,
        1.0f);

    const uint3 quantized_centroid = min(
        convert_uint3(normalized_centroid * 1024.0f),
        (uint3)(1023u));

    const MortonCode morton_code = (expand_morton_bits(quantized_centroid.x) << 2)
        | (expand_morton_bits(quantized_centroid.y) << 1)
        | expand_morton_bits(quantized_centroid.z);

    morton_pairs[face_index] = (uint2)(morton_code, face_index);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sort_morton_pairs(
    __global const uint2* input_pairs,
    __global uint2* output_pairs,
    uint sorted_run_length,
    uint pair_count)
{
    const uint merged_run_length = 2 * sorted_run_length;
    const uint values_per_work_item = min(8u, merged_run_length);
    const uint output_begin = get_global_id(0) * values_per_work_item;

    if (output_begin >= pair_count) {
        return;
    }

    const uint run_begin = output_begin / merged_run_length * merged_run_length;
    const uint right_run_begin = min(run_begin + sorted_run_length, pair_count);
    const uint run_end = min(run_begin + merged_run_length, pair_count);
    const uint diagonal_index = output_begin - run_begin;
    const uint left_count = right_run_begin - run_begin;
    const uint right_count = run_end - right_run_begin;

    uint partition_begin = diagonal_index > right_count
        ? diagonal_index - right_count
        : 0;
    uint partition_end = min(diagonal_index, left_count);

    while (partition_begin < partition_end) {
        const uint partition_middle = partition_begin + (partition_end - partition_begin) / 2;
        const uint left_index = run_begin + partition_middle;
        const uint right_index = right_run_begin + diagonal_index - partition_middle - 1;
        const bool left_precedes_right = input_pairs[left_index].x <= input_pairs[right_index].x;

        if (left_precedes_right) {
            partition_begin = partition_middle + 1;
        } else {
            partition_end = partition_middle;
        }
    }

    uint left_index = run_begin + partition_begin;
    uint right_index = right_run_begin + diagonal_index - partition_begin;
    const uint output_end = min(output_begin + values_per_work_item, run_end);

    for (uint output_index = output_begin; output_index < output_end; ++output_index) {
        const bool take_left = left_index < right_run_begin &&
            (right_index == run_end || input_pairs[left_index].x <= input_pairs[right_index].x);

        if (take_left) {
            output_pairs[output_index] = input_pairs[left_index++];
        } else {
            output_pairs[output_index] = input_pairs[right_index++];
        }
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void init_bvh_leaves(
    __global const uint2* morton_pairs,
    __global const AABBGPU* triangle_bounding_boxes,
    __global BVHNodeGPU* nodes,
    __global uint* leaf_triangle_indices,
    __global uint* parent_indices,
    uint face_count)
{
    const uint primitive_index = get_global_id(0);

    if (primitive_index >= face_count) {
        return;
    }

    const uint face_index = morton_pairs[primitive_index].y;
    const uint leaf_index = face_count - 1 + primitive_index;
    const BVHNodeGPU leaf = {
        triangle_bounding_boxes[face_index],
        ~0u,
        ~0u,
    };

    nodes[leaf_index] = leaf;
    leaf_triangle_indices[primitive_index] = face_index;

    if (primitive_index == 0) {
        parent_indices[0] = ~0u;
    }
}

static inline int common_prefix_length(
    __global const uint2* morton_pairs,
    uint pair_count,
    int first_index,
    int second_index)
{
    const bool second_index_outside = second_index < 0 || (uint)second_index >= pair_count;

    if (second_index_outside) {
        return -1;
    }

    const MortonCode first_code = morton_pairs[first_index].x;
    const MortonCode second_code = morton_pairs[second_index].x;

    if (first_code == second_code) {
        return 32 + clz((uint)(first_index ^ second_index));
    }

    return clz(first_code ^ second_code);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void build_bvh_topology(
    __global const uint2* morton_pairs,
    __global BVHNodeGPU* nodes,
    __global uint* parent_indices,
    uint face_count)
{
    const int node_index = get_global_id(0);
    const bool is_internal_node = face_count > 1 && (uint)node_index < face_count - 1;

    if (!is_internal_node) {
        return;
    }

    const int left_prefix_length = common_prefix_length(
        morton_pairs,
        face_count,
        node_index,
        node_index - 1);

    const int right_prefix_length = common_prefix_length(
        morton_pairs,
        face_count,
        node_index,
        node_index + 1);

    const int direction = right_prefix_length > left_prefix_length ? 1 : -1;

    const int minimum_prefix_length = common_prefix_length(
        morton_pairs,
        face_count,
        node_index,
        node_index - direction);

    int maximum_range_length = 2;

    while (true) {
        const int candidate_index = node_index + maximum_range_length * direction;
        const int candidate_prefix_length = common_prefix_length(
            morton_pairs,
            face_count,
            node_index,
            candidate_index);

        if (candidate_prefix_length <= minimum_prefix_length) {
            break;
        }

        maximum_range_length *= 2;
    }

    int range_length = 0;

    for (int stride = maximum_range_length / 2; stride > 0; stride /= 2) {
        const int candidate_index = node_index + (range_length + stride) * direction;
        const int candidate_prefix_length = common_prefix_length(
            morton_pairs,
            face_count,
            node_index,
            candidate_index);

        if (candidate_prefix_length > minimum_prefix_length) {
            range_length += stride;
        }
    }

    const int range_end = node_index + range_length * direction;
    const int first_primitive_index = min(node_index, range_end);
    const int last_primitive_index = max(node_index, range_end);

    const int range_prefix_length = common_prefix_length(
        morton_pairs,
        face_count,
        first_primitive_index,
        last_primitive_index);

    int split_index = first_primitive_index;
    int stride = last_primitive_index - first_primitive_index;

    do {
        stride = (stride + 1) / 2;
        const int candidate_index = split_index + stride;

        if (candidate_index < last_primitive_index) {
            const int candidate_prefix_length = common_prefix_length(
                morton_pairs,
                face_count,
                first_primitive_index,
                candidate_index);

            if (candidate_prefix_length > range_prefix_length) {
                split_index = candidate_index;
            }
        }
    } while (stride > 1);

    const uint first_leaf_index = face_count - 1;

    const uint left_child_index = split_index == first_primitive_index
        ? first_leaf_index + split_index
        : split_index;

    const uint right_child_index = split_index + 1 == last_primitive_index
        ? first_leaf_index + split_index + 1
        : split_index + 1;

    nodes[node_index].leftChildIndex = left_child_index;
    nodes[node_index].rightChildIndex = right_child_index;

    parent_indices[left_child_index] = node_index;
    parent_indices[right_child_index] = node_index;
}
