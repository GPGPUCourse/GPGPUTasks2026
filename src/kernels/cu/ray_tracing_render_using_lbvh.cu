#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#endif
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

// BVH traversal: closest hit along ray
__device__ bool bvh_closest_hit(
    const float3& orig,
    const float3& dir,
    const BVHNodeGPU* nodes,
    const unsigned int* leafTriIndices,
    unsigned int nfaces,
    const float* vertices,
    const unsigned int* faces,
    float tMin,
    float& outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    int& outFaceId,
    float& outU, // сюда нужно записать u рассчитанный в intersect_ray_triangle(..., t, u, v)
    float& outV) // сюда нужно записать v рассчитанный в intersect_ray_triangle(..., t, u, v)
{
    const int rootIndex = 0;
    const int stack_depth = 34;
    const int leafStart = (int)nfaces - 1;

    int dfs_stack[stack_depth];
    int stack_head = 0;

    int curr_node_num = rootIndex;

    int min_face_num = -1;
    float min_t = FLT_MAX;
    float min_u;
    float min_v;

    while(stack_head > 0  || curr_node_num >= 0){
        if(curr_node_num < 0){
            stack_head--;
            curr_node_num = dfs_stack[stack_head];
            dfs_stack[stack_head] = -1;
        }
        else if(curr_node_num < leafStart){
            BVHNodeGPU node = nodes[curr_node_num];
            float t_near, t_far;
            if(intersect_ray_aabb_any(orig, dir, node.aabb, t_near, t_far)){
                curr_node_num = node.leftChildIndex;
                curassert(stack_head < stack_depth, 2103893874);
                dfs_stack[stack_head] = node.rightChildIndex;
                stack_head++;
            }
            else{
                curr_node_num = -1;
            }
        }
        else{
            float curr_t, curr_u, curr_v;
            int curr_face_num = leafTriIndices[curr_node_num - leafStart];
            uint3 face = loadFace(faces, curr_face_num);
            float3 v0, v1, v2;
            v0 = loadVertex(vertices, face.x);
            v1 = loadVertex(vertices, face.y);
            v2 = loadVertex(vertices, face.z);
            if(intersect_ray_triangle_any(orig, dir, v0, v1, v2, false, curr_t, curr_u, curr_v)){
                if(curr_t < min_t){
                    min_face_num = curr_face_num;
                    min_t = curr_t;
                    min_u = curr_u;
                    min_v = curr_v;
                }
            }

            curr_node_num = -1;
        }
    }

    if(min_face_num >= 0){
        outT = min_t;
        outU = min_u;
        outV = min_v;
        outFaceId = min_face_num;
        return true;
    }
    return false; // no intersections found
}

// BVH traversal: any hit (for AO rays)
// ignore_face: do not count intersections with this triangle index
__device__ bool any_hit_from(
    const float3& orig,
    const float3& dir,
    const float* vertices,
    const unsigned int* faces,
    const BVHNodeGPU* nodes,
    const unsigned int* leafTriIndices,
    unsigned int nfaces,
    int ignore_face)
{
    const int rootIndex = 0;
    const int stack_depth = 34;
    const int leafStart = (int)nfaces - 1;

    int dfs_stack[stack_depth];
    int stack_head = 0;

    int curr_node_num = rootIndex;

    while(stack_head > 0  || curr_node_num >= 0){
        if(curr_node_num < 0){
            stack_head--;
            curr_node_num = dfs_stack[stack_head];
            dfs_stack[stack_head] = -1;
        }
        else if(curr_node_num < leafStart){
            BVHNodeGPU node = nodes[curr_node_num];
            float t_near, t_far;
            if(intersect_ray_aabb_any(orig, dir, node.aabb, t_near, t_far)){
                curr_node_num = node.leftChildIndex;
                curassert(stack_head < stack_depth, 2103893874);
                dfs_stack[stack_head] = node.rightChildIndex;
                stack_head++;
            }
            else{
                curr_node_num = -1;
            }
        }
        else{
            float curr_t, curr_u, curr_v;
            int curr_face_num = leafTriIndices[curr_node_num - leafStart];
            if(curr_face_num == ignore_face){
                curr_node_num = -1;
                continue;
            }
            uint3 face = loadFace(faces, curr_face_num);
            float3 v0, v1, v2;
            v0 = loadVertex(vertices, face.x);
            v1 = loadVertex(vertices, face.y);
            v2 = loadVertex(vertices, face.z);
            if(intersect_ray_triangle_any(orig, dir, v0, v1, v2, false, curr_t, curr_u, curr_v)){
                return true;
            }

            curr_node_num = -1;
        }
    }
    return false; // no intersections found
}

// + helper: build tangent basis for a given normal
__device__ inline void make_basis(const float3& n, float3& t, float3& b) {
    // pick a non-parallel vector
    float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.f,0.f,1.f) : make_float3(0.f,1.f,0.f);
    t = normalize_f3(cross_f3(up, n));
    b = cross_f3(n, t);
}

__global__ void ray_tracing_render_using_lbvh(
    const float* vertices,
    const unsigned int* faces,
    const BVHNodeGPU* bvhNodes,
    const unsigned int* leafTriIndices,
    int* framebuffer_face_id,
    float* framebuffer_ambient_occlusion,
    CameraViewGPU *camera,
    unsigned int nfaces
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera.magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 946435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

    float tMin  = 1e-6f;
    float tBest = FLT_MAX;
    float uBest=0, vBest=0;
    int   faceIdBest = -1;

    // Use BVH traversal instead of brute-force loop
    bool hit = bvh_closest_hit(
        ray_origin,
        ray_direction,
        bvhNodes,
        leafTriIndices,
        nfaces,
        vertices,
        faces,
        tMin,
        tBest,
        faceIdBest,
        uBest,
        vBest
    );

    const unsigned int idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = faceIdBest;

    float ao = 1.0f; // background stays white
    if (faceIdBest >= 0) {
        uint3 f = loadFace(faces, faceIdBest);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
        float3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        float3 n  = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x*ray_direction.x + n.y*ray_direction.y + n.z*ray_direction.z > 0.0f)
            n = make_float3(-n.x, -n.y, -n.z);

        float3 P  = {ray_origin.x + tBest*ray_direction.x,
            ray_origin.y + tBest*ray_direction.y,
            ray_origin.z + tBest*ray_direction.z};

        float scale = fmaxf(fmaxf(length_f3(e1), length_f3(e2)),
            length_f3(make_float3(c.x-a.x, c.y-a.y, c.z-a.z)));
        float eps   = 1e-3f * fmaxf(1.0f, scale);
        float3 Po   = {P.x + n.x*eps, P.y + n.y*eps, P.z + n.z*eps};

        // build tangent basis
        float3 T, B;
        make_basis(n, T, B);
        // per-pixel seed (stable)
        union {
            float f32;
            uint32_t u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint32_t rng = 0x9E3779B9u ^ idx ^ tBestUnion.u32;

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1 = random01(rng);
            float u2 = random01(rng);
            float z  = u1;                          // z in [0,1]
            float phi = 6.28318530718f * u2;        // 2*pi*u2
            float r  = sqrtf(fmaxf(0.f, 1.f - z*z));
            float3 d_local = make_float3(r * cosf(phi), r * sinf(phi), z);

            // transform to world space
            float3 d = make_float3(
                T.x*d_local.x + B.x*d_local.y + n.x*d_local.z,
                T.y*d_local.x + B.y*d_local.y + n.y*d_local.z,
                T.z*d_local.x + B.z*d_local.y + n.z*d_local.z
            );

            if (any_hit_from(Po, d,
                             vertices, faces,
                             bvhNodes, leafTriIndices,
                             nfaces, faceIdBest))
                ++hits;
        }
        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }
    framebuffer_ambient_occlusion[idx] = ao;
}


namespace cuda {
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::ray_tracing_render_using_lbvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        bvhNodes.cuptr(), leafTriIndices.cuptr(),
        framebuffer_face_id.cuptr(), framebuffer_ambient_occlusion.cuptr(),
        camera.cuptr(),
        nfaces
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
