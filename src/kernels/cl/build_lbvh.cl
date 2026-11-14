#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
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
