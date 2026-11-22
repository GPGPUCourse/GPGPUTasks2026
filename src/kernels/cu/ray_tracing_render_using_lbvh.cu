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
#include "ambient_occlusion.cuh"

// stack-based BVH traversal and shortening the ray interval after a closer hit. https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
// https://github.com/jbikker/bvh_article/blob/a4aaed059dddeca8c134b4511caf9d17202af3db/bvh.cpp#L118-L154
static __device__ int trace_bvh(float3 origin, float3 direction, const BVHNodeGPU* nodes,
    const unsigned* leafFaces, int faceCount, const float* vertices, const unsigned* faces,
    float tMin, float& tBest, int ignoreFace = -1)
{
    int stack[64];
    stack[0] = 0;
    int size = 1;
    int bestFace = -1;
    while (size > 0) {
        int index = stack[--size];
        const BVHNodeGPU& node = nodes[index];
        float nearT, farT;
        // https://github.com/jbikker/bvh_article/blob/a4aaed059dddeca8c134b4511caf9d17202af3db/bvh.cpp#L16-L54
        if (!intersect_ray_aabb(origin, direction, node.aabb, tMin, tBest, nearT, farT))
            continue;
        if (index < faceCount - 1) {
            stack[size++] = node.rightChildIndex;
            stack[size++] = node.leftChildIndex;
            continue;
        }
        int face = leafFaces[index - (faceCount - 1)];
        if (face == ignoreFace)
            continue;
        uint3 f = loadFace(faces, face);
        float t, u, v;
        if (intersect_ray_triangle(origin, direction, loadVertex(vertices, f.x),
                loadVertex(vertices, f.y), loadVertex(vertices, f.z), tMin, tBest, false, t, u, v)) {
            if (ignoreFace >= 0)
                return face;
            if (t < tBest || face > bestFace) {
                tBest = t;
                bestFace = face;
            }
        }
    }
    return bestFace;
}

__global__ void ray_tracing_render_using_lbvh(
    const float* vertices,
    const unsigned int* faces,
    const BVHNodeGPU* bvhNodes,
    const unsigned int* leafTriIndices,
    int* framebuffer_face_id,
    float* framebuffer_ambient_occlusion,
    CameraViewGPU *camera,
    unsigned int nfaces, float* depth
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera->magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 946435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

    float tMin  = 1e-6f;
    float tBest = FLT_MAX;
    int faceIdBest = trace_bvh(ray_origin, ray_direction, bvhNodes, leafTriIndices,
                               nfaces, vertices, faces, tMin, tBest);

    const unsigned int idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = faceIdBest;
    if (depth) depth[idx] = tBest;

    framebuffer_ambient_occlusion[idx] = ambient_occlusion(vertices, faces, ray_origin, ray_direction,
        faceIdBest, tBest, idx, [&](float3 origin, float3 direction, int face) {
            float tMax = FLT_MAX;
            return trace_bvh(origin, direction, bvhNodes, leafTriIndices, nfaces, vertices, faces,
                             1e-4f, tMax, face) >= 0;
        });
}


namespace cuda {
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces, gpu::gpu_mem_32f* depth)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::ray_tracing_render_using_lbvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        bvhNodes.cuptr(), leafTriIndices.cuptr(),
        framebuffer_face_id.cuptr(), framebuffer_ambient_occlusion.cuptr(),
        camera.cuptr(),
        nfaces, depth ? depth->cuptr() : nullptr
        );
    CUDA_CHECK_KERNEL_SYNC(stream);
}
} // namespace cuda

#include "build_lbvh.cuh"
#include "denoise_ao.cuh"
#include "optix_renderer.cuh"
