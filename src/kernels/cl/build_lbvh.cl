#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
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
