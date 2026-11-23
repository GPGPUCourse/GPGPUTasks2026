#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

__global__ void build_morton_codes(const float* vertices, const unsigned int* faces, float lo, float size,
    unsigned int* codes, unsigned int* indices, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float3 a = cuda::fetch_float3(vertices, faces[3 * i]);
    float3 b = cuda::fetch_float3(vertices, faces[3 * i + 1]);
    float3 c = cuda::fetch_float3(vertices, faces[3 * i + 2]);
    unsigned int x = min(max((int)(((a.x + b.x + c.x) / 3 - lo) / size * 1024), 0), 1023);
    unsigned int y = min(max((int)(((a.y + b.y + c.y) / 3 - lo) / size * 1024), 0), 1023);
    unsigned int z = min(max((int)(((a.z + b.z + c.z) / 3 - lo) / size * 1024), 0), 1023);
    unsigned int code = 0;
    for (int bit = 0; bit < 10; ++bit) {
        code |= ((x >> bit) & 1u) << (3 * bit + 2);
        code |= ((y >> bit) & 1u) << (3 * bit + 1);
        code |= ((z >> bit) & 1u) << (3 * bit);
    }
    codes[i] = code;
    indices[i] = i;
}

__device__ int common_prefix(const unsigned int* codes, int n, int i, int j)
{
    if (j < 0 || j >= n) {
        return -1;
    }
    if (codes[i] == codes[j]) {
        return 32 + __clz((unsigned int)(i ^ j));
    }
    return __clz(codes[i] ^ codes[j]);
}

__global__ void build_nodes(const float* vertices, const unsigned int* faces,
    const unsigned int* codes, const unsigned int* indices,
    BVHNodeGPU* nodes, unsigned int* levels, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const int fi = indices[i];
    float3 a = cuda::fetch_float3(vertices, faces[3 * fi]);
    float3 b = cuda::fetch_float3(vertices, faces[3 * fi + 1]);
    float3 c = cuda::fetch_float3(vertices, faces[3 * fi + 2]);
    nodes[n - 1 + i].aabb = {
        fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z)),
        fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z))};
    nodes[n - 1 + i].leftChildIndex = 0xffffffff;
    nodes[n - 1 + i].rightChildIndex = 0xffffffff;
    if (i >= n - 1) {
        return;
    }

    int d = common_prefix(codes, n, i, i + 1) > common_prefix(codes, n, i, i - 1) ? 1 : -1;
    int prefix = common_prefix(codes, n, i, i - d);
    int len = 1;
    int end = d == 1 ? n - i : i + 1;
    while (len + 1 < end) {
        int mid = len + (end - len) / 2;
        if (common_prefix(codes, n, i, i + mid * d) > prefix) {
            len = mid;
        } else {
            end = mid;
        }
    }

    int first = min(i, i + len * d);
    int last = max(i, i + len * d);
    prefix = common_prefix(codes, n, first, last);
    int left = first;
    int right = last;
    while (left + 1 < right) {
        int mid = left + (right - left) / 2;
        if (common_prefix(codes, n, first, mid) > prefix) {
            left = mid;
        } else {
            right = mid;
        }
    }

    int a_index = left == first ? n - 1 + left : left;
    int b_index = left + 1 == last ? n + left : left + 1;
    nodes[i].leftChildIndex = a_index;
    nodes[i].rightChildIndex = b_index;
    levels[i] = prefix;
}

__global__ void build_bounds(BVHNodeGPU* nodes, const unsigned int* levels, int level, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1 || levels[i] != level) {
        return;
    }

    const AABBGPU& a = nodes[nodes[i].leftChildIndex].aabb;
    const AABBGPU& b = nodes[nodes[i].rightChildIndex].aabb;
    nodes[i].aabb = {fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y), fminf(a.min_z, b.min_z),
                     fmaxf(a.max_x, b.max_x), fmaxf(a.max_y, b.max_y), fmaxf(a.max_z, b.max_z)};
}

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
    const int leafStart = (int)nfaces - 1;

    int stack[64];
    int size = 0;
    stack[size++] = rootIndex;
    bool hit = false;
    while (size > 0) {
        const int i = stack[--size];
        const BVHNodeGPU& node = nodes[i];
        float near, far;
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, outT, near, far)) {
            continue;
        }
        if (i < leafStart) {
            stack[size++] = node.leftChildIndex;
            stack[size++] = node.rightChildIndex;
        } else {
            const int fi = leafTriIndices[i - leafStart];
            uint3 f = loadFace(faces, fi);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);
            float t, u, v;
            if (intersect_ray_triangle(orig, dir, a, b, c, tMin, outT, false, t, u, v)) {
                outT = t;
                outFaceId = fi;
                outU = u;
                outV = v;
                hit = true;
            }
        }
    }
    return hit;
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
    const int leafStart = (int)nfaces - 1;

    int stack[64];
    int size = 0;
    stack[size++] = rootIndex;
    while (size > 0) {
        const int i = stack[--size];
        const BVHNodeGPU& node = nodes[i];
        float near, far;
        if (!intersect_ray_aabb(orig, dir, node.aabb, 1e-4f, FLT_MAX, near, far)) {
            continue;
        }
        if (i < leafStart) {
            stack[size++] = node.leftChildIndex;
            stack[size++] = node.rightChildIndex;
        } else {
            const int fi = leafTriIndices[i - leafStart];
            if (fi == ignore_face) {
                continue;
            }
            uint3 f = loadFace(faces, fi);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);
            float t, u, v;
            if (intersect_ray_triangle(orig, dir, a, b, c, 1e-4f, FLT_MAX, false, t, u, v)) {
                return true;
            }
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
void build_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &indices, unsigned int n)
{
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();

    gpu::gpu_mem_32u codes(n), levels(n);
    auto policy = thrust::cuda::par.on(stream);
    auto first = thrust::device_pointer_cast(vertices.cuptr());
    auto bounds = thrust::minmax_element(policy, first, first + vertices.number());
    float lo = *bounds.first, hi = *bounds.second;
    ::build_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), lo, fmaxf(hi - lo, 1e-9f), codes.cuptr(), indices.cuptr(), n);
    thrust::stable_sort_by_key(policy, codes.cuptr(), codes.cuptr() + n, indices.cuptr());
    ::build_nodes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), codes.cuptr(), indices.cuptr(), nodes.cuptr(), levels.cuptr(), n);
    for (int level = 63; level >= 0; --level) {
        ::build_bounds<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(nodes.cuptr(), levels.cuptr(), level, n);
    }
    CUDA_CHECK_KERNEL(stream);
}

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
