#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    const float3 ray_origin,
    const float3 ray_direction,
    __global const BVHNodeGPU* nodes,
    __global const uint* leaf_triangle_indices,
    uint face_count,
    __global const float* vertices,
    __global const uint* faces,
    float minimum_hit_distance,
    __private float* closest_hit_distance,
    __private int* closest_face_id,
    __private float* closest_barycentric_u,
    __private float* closest_barycentric_v)
{
    const int root_index = 0;
    const int first_leaf_index = (int)face_count - 1;

    if (face_count == 0) {
        return false;
    }

    int node_stack[BVH_MAX_DEPTH];
    int stack_size = 0;
    node_stack[stack_size++] = root_index;

    while (stack_size > 0) {
        const int node_index = node_stack[--stack_size];
        const BVHNodeGPU node = nodes[node_index];
        float box_entry_distance, box_exit_distance;

        const bool intersects_node_bounds = intersect_ray_aabb(
            ray_origin,
            ray_direction,
            node.aabb,
            minimum_hit_distance,
            *closest_hit_distance,
            &box_entry_distance,
            &box_exit_distance);

        if (!intersects_node_bounds) {
            continue;
        }

        if (node_index >= first_leaf_index) {
            const int face_id = leaf_triangle_indices[node_index - first_leaf_index];
            const uint3 vertex_indices = loadFace(faces, face_id);
            float hit_distance, barycentric_u, barycentric_v;

            const bool intersects_triangle = intersect_ray_triangle(
                ray_origin,
                ray_direction,
                loadVertex(vertices, vertex_indices.x),
                loadVertex(vertices, vertex_indices.y),
                loadVertex(vertices, vertex_indices.z),
                minimum_hit_distance,
                *closest_hit_distance,
                false,
                &hit_distance,
                &barycentric_u,
                &barycentric_v);

            if (!intersects_triangle) {
                continue;
            }

            const bool is_closer_hit = hit_distance < *closest_hit_distance;
            const bool is_preferred_equal_hit = hit_distance == *closest_hit_distance &&
                face_id > *closest_face_id;

            if (is_closer_hit || is_preferred_equal_hit) {
                *closest_hit_distance = hit_distance;
                *closest_face_id = face_id;
                *closest_barycentric_u = barycentric_u;
                *closest_barycentric_v = barycentric_v;
            }
        } else {
            rassert(stack_size + 2 <= BVH_MAX_DEPTH, 873456212);
            if (stack_size + 2 > BVH_MAX_DEPTH) {
                return false;
            }
            node_stack[stack_size++] = node.rightChildIndex;
            node_stack[stack_size++] = node.leftChildIndex;
        }
    }

    return *closest_face_id >= 0;
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    const float3 ray_origin,
    const float3 ray_direction,
    __global const float* vertices,
    __global const uint* faces,
    __global const BVHNodeGPU* nodes,
    __global const uint* leaf_triangle_indices,
    uint face_count,
    int ignored_face_id)
{
    const int root_index = 0;
    const int first_leaf_index = (int)face_count - 1;
    const float minimum_hit_distance = 1e-4f;

    if (face_count == 0) {
        return false;
    }

    int node_stack[BVH_MAX_DEPTH];
    int stack_size = 0;
    node_stack[stack_size++] = root_index;

    while (stack_size > 0) {
        const int node_index = node_stack[--stack_size];
        const BVHNodeGPU node = nodes[node_index];
        float box_entry_distance, box_exit_distance;

        const bool intersects_node_bounds = intersect_ray_aabb(
            ray_origin,
            ray_direction,
            node.aabb,
            minimum_hit_distance,
            FLT_MAX,
            &box_entry_distance,
            &box_exit_distance);

        if (!intersects_node_bounds) {
            continue;
        }

        if (node_index >= first_leaf_index) {
            const int face_id = leaf_triangle_indices[node_index - first_leaf_index];

            if (face_id == ignored_face_id) {
                continue;
            }

            const uint3 vertex_indices = loadFace(faces, face_id);
            float hit_distance, barycentric_u, barycentric_v;

            const bool intersects_triangle = intersect_ray_triangle(
                ray_origin,
                ray_direction,
                loadVertex(vertices, vertex_indices.x),
                loadVertex(vertices, vertex_indices.y),
                loadVertex(vertices, vertex_indices.z),
                minimum_hit_distance,
                FLT_MAX,
                false,
                &hit_distance,
                &barycentric_u,
                &barycentric_v);

            if (intersects_triangle) {
                return true;
            }
        } else {
            rassert(stack_size + 2 <= BVH_MAX_DEPTH, 873456213);

            if (stack_size + 2 > BVH_MAX_DEPTH) {
                return false;
            }

            node_stack[stack_size++] = node.rightChildIndex;
            node_stack[stack_size++] = node.leftChildIndex;
        }
    }

    return false;
}

// Helper: build tangent basis for a given normal
static inline void make_basis(const float3 n,
    __private float3* t,
    __private float3* b)
{
    // pick a non-parallel vector
    float3 up = fabs(n.z) < 0.999f
        ? (float3)(0.0f, 0.0f, 1.0f)
        : (float3)(0.0f, 1.0f, 0.0f);

    *t = normalize_f3(cross_f3(up, n));
    *b = cross_f3(n, *t);
}

__kernel void ray_tracing_render_using_lbvh(
    __global const float* vertices,
    __global const uint* faces,
    __global const BVHNodeGPU* bvhNodes,
    __global const uint* leafTriIndices,
    __global int* framebuffer_face_id,
    __global float* framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
    uint nfaces)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    rassert(camera->magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 786435342);
    if (i >= camera->K.width || j >= camera->K.height) {
        return;
    }

    float3 ray_origin;
    float3 ray_direction;

    make_primary_ray(
        camera,
        (float)i + 0.5f,
        (float)j + 0.5f,
        &ray_origin,
        &ray_direction);

    float minimum_hit_distance = 1e-6f;
    float closest_hit_distance = FLT_MAX;
    float closest_barycentric_u = 0.0f;
    float closest_barycentric_v = 0.0f;
    int closest_face_id = -1;

    // Use BVH traversal instead of brute-force loop
    bool hit = bvh_closest_hit(
        ray_origin,
        ray_direction,
        bvhNodes,
        leafTriIndices,
        nfaces,
        vertices,
        faces,
        minimum_hit_distance,
        &closest_hit_distance,
        &closest_face_id,
        &closest_barycentric_u,
        &closest_barycentric_v);

    const uint idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = closest_face_id;

    float ao = 1.0f; // background stays white

    if (closest_face_id >= 0) {
        uint3 f = loadFace(faces, (uint)closest_face_id);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = (float3)(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 e2 = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float3 n = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) {
            n = (float3)(-n.x, -n.y, -n.z);
        }

        float3 P = (float3)(
            ray_origin.x + closest_hit_distance * ray_direction.x,
            ray_origin.y + closest_hit_distance * ray_direction.y,
            ray_origin.z + closest_hit_distance * ray_direction.z);

        float3 ac = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float scale = fmax(
            fmax(length_f3(e1), length_f3(e2)),
            length_f3(ac));

        float eps = 1e-3f * fmax(1.0f, scale);
        float3 Po = (float3)(
            P.x + n.x * eps,
            P.y + n.y * eps,
            P.z + n.z * eps);

        // build tangent basis
        float3 T;
        float3 B;
        make_basis(n, &T, &B);

        // per-pixel seed (stable)
        union {
            float f32;
            uint u32;
        } hit_distance_bits;
        hit_distance_bits.f32 = closest_hit_distance;
        uint rng = (uint)(0x9E3779B9u ^ idx ^ hit_distance_bits.u32);

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1 = random01(&rng);
            float u2 = random01(&rng);
            float z = u1; // z in [0,1]
            float phi = 6.28318530718f * u2; // 2*pi*u2
            float r = sqrt(fmax(0.0f, 1.0f - z * z));
            float3 d_local = (float3)(
                r * cos(phi),
                r * sin(phi),
                z);

            // transform to world space
            float3 d = (float3)(
                T.x * d_local.x + B.x * d_local.y + n.x * d_local.z,
                T.y * d_local.x + B.y * d_local.y + n.y * d_local.z,
                T.z * d_local.x + B.z * d_local.y + n.z * d_local.z);

            const bool is_occluded = any_hit_from(
                Po, d,
                vertices, faces,
                bvhNodes, leafTriIndices,
                nfaces, closest_face_id);

            if (is_occluded) {
                ++hits;
            }
        }

        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }

    framebuffer_ambient_occlusion[idx] = ao;
}
