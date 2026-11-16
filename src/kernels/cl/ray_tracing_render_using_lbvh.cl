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

// A nonnegative ignoreFace selects any-hit mode for an AO ray.
static inline bool trace_bvh(
    const float3              orig,
    const float3              dir,
    __global const BVHNodeGPU* nodes,
    __global const uint*      leafTriIndices,
    uint                      nfaces,
    __global const float*     vertices,
    __global const uint*      faces,
    float                     tMin,
    int                       ignoreFace,
    __private float*          outT,
    __private int*            outFaceId,
    __private float*          outU,
    __private float*          outV)
{
    if (nfaces == 0)
        return false;

    const uint leafStart = nfaces - 1;
    // Each split consumes at least one bit of the 30-bit Morton code
    // or the 32-bit tie breaker. Keep at most one sibling per level.
    uint stack[64];
    uint pending = 0;
    uint nodeIndex = 0;
    float bestT = *outT;
    int bestFace = -1;

    float nearT, farT;
    if (!intersect_ray_aabb(orig, dir, nodes[0].aabb, tMin, bestT, &nearT, &farT))
        return false;

    for (;;) {
        if (nodeIndex >= leafStart) {
            const int faceId = (int)leafTriIndices[nodeIndex - leafStart];
            if (faceId != ignoreFace) {
                const uint3 f = loadFace(faces, (uint)faceId);
                float t, u, v;
                if (intersect_ray_triangle(orig, dir,
                                           loadVertex(vertices, f.x),
                                           loadVertex(vertices, f.y),
                                           loadVertex(vertices, f.z),
                                           tMin, bestT, false, &t, &u, &v)) {
                    if (ignoreFace >= 0)
                        return true;
                    // Increasing face IDs win exact distance ties, as in brute force.
                    if (bestFace < 0 || t < bestT || faceId > bestFace) {
                        bestT = t;
                        bestFace = faceId;
                        *outU = u;
                        *outV = v;
                    }
                }
            }
        } else {
            const uint left = nodes[nodeIndex].leftChildIndex;
            const uint right = nodes[nodeIndex].rightChildIndex;
            float leftNear, rightNear;
            const bool hitLeft = intersect_ray_aabb(orig, dir, nodes[left].aabb,
                                                    tMin, bestT, &leftNear, &farT);
            const bool hitRight = intersect_ray_aabb(orig, dir, nodes[right].aabb,
                                                     tMin, bestT, &rightNear, &farT);
            if (hitLeft && hitRight) {
                const bool leftFirst = leftNear <= rightNear;
                stack[pending++] = leftFirst ? right : left;
                nodeIndex = leftFirst ? left : right;
                continue;
            }
            if (hitLeft || hitRight) {
                nodeIndex = hitLeft ? left : right;
                continue;
            }
        }

        // A closer triangle may have shortened the ray since a node was queued.
        do {
            if (pending == 0) {
                *outT = bestT;
                *outFaceId = bestFace;
                return bestFace >= 0;
            }
            nodeIndex = stack[--pending];
        } while (!intersect_ray_aabb(orig, dir, nodes[nodeIndex].aabb,
                                    tMin, bestT, &nearT, &farT));
    }
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    const float3              orig,
    const float3              dir,
    __global const float*     vertices,
    __global const uint*      faces,
    __global const BVHNodeGPU* nodes,
    __global const uint*      leafTriIndices,
    uint                      nfaces,
    int                       ignore_face)
{
    float t = FLT_MAX, u, v;
    int faceId;
    return trace_bvh(orig, dir, nodes, leafTriIndices, nfaces, vertices, faces,
                     1e-4f, ignore_face, &t, &faceId, &u, &v);
}

// Helper: build tangent basis for a given normal
static inline void make_basis(const float3 n,
                              __private float3* t,
                              __private float3* b)
{
    // pick a non-parallel vector
    float3 up = (fabs(n.z) < 0.999f)
        ? (float3)(0.0f, 0.0f, 1.0f)
        : (float3)(0.0f, 1.0f, 0.0f);

    *t = normalize_f3(cross_f3(up, n));
    *b = cross_f3(n, *t);
}

__kernel void ray_tracing_render_using_lbvh(
    __global const float*      vertices,
    __global const uint*       faces,
    __global const BVHNodeGPU* bvhNodes,
    __global const uint*       leafTriIndices,
    __global int*              framebuffer_face_id,
    __global float*            framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
    uint                       nfaces)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    rassert(camera->magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 786435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin;
    float3 ray_direction;
    make_primary_ray(camera,
                     (float)i + 0.5f,
                     (float)j + 0.5f,
                     &ray_origin,
                     &ray_direction);

    float tMin      = 1e-6f;
    float tBest     = FLT_MAX;
    float uBest     = 0.0f;
    float vBest     = 0.0f;
    int   faceIdBest = -1;

    // Use BVH traversal instead of brute-force loop
    trace_bvh(
        ray_origin,
        ray_direction,
        bvhNodes,
        leafTriIndices,
        nfaces,
        vertices,
        faces,
        tMin,
        -1,
        &tBest,
        &faceIdBest,
        &uBest,
        &vBest);

    const uint idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = faceIdBest;

    float ao = 1.0f; // background stays white

    if (faceIdBest >= 0) {
        uint3  f = loadFace(faces, (uint)faceIdBest);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = (float3)(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 e2 = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float3 n  = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x * ray_direction.x +
            n.y * ray_direction.y +
            n.z * ray_direction.z > 0.0f)
        {
            n = (float3)(-n.x, -n.y, -n.z);
        }

        float3 P = (float3)(ray_origin.x + tBest * ray_direction.x,
                            ray_origin.y + tBest * ray_direction.y,
                            ray_origin.z + tBest * ray_direction.z);

        float3 ac = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float  scale = fmax(fmax(length_f3(e1), length_f3(e2)),
                            length_f3(ac));

        float  eps = 1e-3f * fmax(1.0f, scale);
        float3 Po  = (float3)(P.x + n.x * eps,
                              P.y + n.y * eps,
                              P.z + n.z * eps);

        // build tangent basis
        float3 T;
        float3 B;
        make_basis(n, &T, &B);

        // per-pixel seed (stable)
        union {
            float f32;
            uint  u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint rng = (uint)(0x9E3779B9u ^ idx ^ tBestUnion.u32);

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1  = random01(&rng);
            float u2  = random01(&rng);
            float z   = u1;                      // z in [0,1]
            float phi = 6.28318530718f * u2;     // 2*pi*u2
            float r   = sqrt(fmax(0.0f, 1.0f - z * z));
            float3 d_local = (float3)(r * cos(phi),
                                      r * sin(phi),
                                      z);

            // transform to world space
            float3 d = (float3)(
                T.x * d_local.x + B.x * d_local.y + n.x * d_local.z,
                T.y * d_local.x + B.y * d_local.y + n.y * d_local.z,
                T.z * d_local.x + B.z * d_local.y + n.z * d_local.z
            );

            if (any_hit_from(Po, d,
                             vertices, faces,
                             bvhNodes, leafTriIndices,
                             nfaces, faceIdBest))
            {
                ++hits;
            }
        }

        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }

    framebuffer_ambient_occlusion[idx] = ao;
}
