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

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

static const int kFarStack = 64;

__device__ bool walk_lbvh(
    const float3& orig,
    const float3& dir,
    const BVHNodeGPU* nodes,
    const unsigned int* leafTriIndices,
    unsigned int nfaces,
    const float* vertices,
    const unsigned int* faces,
    float tMin,
    float tMax,
    bool anyHit,
    int ignoreFace,
    float& outT,
    int& outFace,
    float& outU,
    float& outV)
{
    if (nfaces == 0)
        return false;

    const int leafStart = (int)nfaces - 1;
    int farChild[kFarStack];
    int sp = 0;
    int idx = 0;
    float bestT = tMax;
    bool found = false;

    for (;;) {
        bool down = true;
        while (down) {
            down = false;
            const BVHNodeGPU node = nodes[idx];
            float tNear, tFar;
            if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, bestT, tNear, tFar))
                break;

            if (idx >= leafStart) {
                const unsigned int fi = leafTriIndices[idx - leafStart];
                if (!(anyHit && (int)fi == ignoreFace)) {
                    const uint3 f = loadFace(faces, fi);
                    const float3 a = loadVertex(vertices, f.x);
                    const float3 b = loadVertex(vertices, f.y);
                    const float3 c = loadVertex(vertices, f.z);
                    float t, u, v;
                    if (intersect_ray_triangle(orig, dir, a, b, c, tMin, bestT, /*backface*/ false, t, u, v)) {
                        if (anyHit)
                            return true;
                        bestT = t;
                        outT = t;
                        outU = u;
                        outV = v;
                        outFace = (int)fi;
                        found = true;
                    }
                }
                break;
            }

            const int left = (int)node.leftChildIndex;
            const int right = (int)node.rightChildIndex;
            float leftNear, leftFar, rightNear, rightFar;
            const bool hitLeft = intersect_ray_aabb(orig, dir, nodes[left].aabb, tMin, bestT, leftNear, leftFar);
            const bool hitRight = intersect_ray_aabb(orig, dir, nodes[right].aabb, tMin, bestT, rightNear, rightFar);
            if (hitLeft && hitRight) {
                const bool leftFirst = leftNear <= rightNear;
                farChild[sp++] = leftFirst ? right : left;
                idx = leftFirst ? left : right;
                down = true;
            } else if (hitLeft || hitRight) {
                idx = hitLeft ? left : right;
                down = true;
            }
        }
        if (sp == 0)
            break;
        idx = farChild[--sp];
    }
    return found;
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
    return walk_lbvh(orig, dir, nodes, leafTriIndices, nfaces, vertices, faces,
        tMin, FLT_MAX, false, -1, outT, outFaceId, outU, outV);
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
    float t, u, v;
    int face = -1;
    return walk_lbvh(orig, dir, nodes, leafTriIndices, nfaces, vertices, faces,
        1e-4f, FLT_MAX, true, ignore_face, t, face, u, v);
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


__device__ int clz32_d(unsigned int x)
{
    return x == 0u ? 32 : __clz(x);
}

__device__ int common_prefix_d(const unsigned int* codes, int count, int i, int j)
{
    if (j < 0 || j >= count)
        return -1;
    const unsigned int ci = codes[i];
    const unsigned int cj = codes[j];
    if (ci == cj)
        return 32 + clz32_d((unsigned int)i ^ (unsigned int)j);
    return clz32_d(ci ^ cj);
}

__device__ void determine_range_d(const unsigned int* codes, int count, int i, int& first, int& last)
{
    const int left = common_prefix_d(codes, count, i, i - 1);
    const int right = common_prefix_d(codes, count, i, i + 1);
    const int direction = (right > left) ? 1 : -1;
    const int minPrefix = common_prefix_d(codes, count, i, i - direction);

    int lengthMax = 2;
    while (common_prefix_d(codes, count, i, i + lengthMax * direction) > minPrefix)
        lengthMax <<= 1;

    int length = 0;
    for (int step = lengthMax >> 1; step > 0; step >>= 1) {
        if (common_prefix_d(codes, count, i, i + (length + step) * direction) > minPrefix)
            length += step;
    }

    const int other = i + length * direction;
    first = other < i ? other : i;
    last = other < i ? i : other;
}

__device__ int find_split_d(const unsigned int* codes, int count, int first, int last)
{
    if (first == last)
        return first;
    const int rangePrefix = common_prefix_d(codes, count, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int next = split + step;
        if (next < last && common_prefix_d(codes, count, first, next) > rangePrefix)
            split = next;
    } while (step > 1);
    return split;
}

__device__ unsigned int expand_bits_d(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ unsigned int morton3D_d(float x, float y, float z)
{
    const int ix = (int)(x * 1024.0f);
    const int iy = (int)(y * 1024.0f);
    const int iz = (int)(z * 1024.0f);
    const unsigned int xx = expand_bits_d((unsigned int)(ix < 0 ? 0 : (ix > 1023 ? 1023 : ix)));
    const unsigned int yy = expand_bits_d((unsigned int)(iy < 0 ? 0 : (iy > 1023 ? 1023 : iy)));
    const unsigned int zz = expand_bits_d((unsigned int)(iz < 0 ? 0 : (iz > 1023 ? 1023 : iz)));
    return (xx << 2) | (yy << 1) | zz;
}

__device__ AABBGPU merge_aabb(const AABBGPU& a, const AABBGPU& b)
{
    AABBGPU box;
    box.min_x = fminf(a.min_x, b.min_x);
    box.min_y = fminf(a.min_y, b.min_y);
    box.min_z = fminf(a.min_z, b.min_z);
    box.max_x = fmaxf(a.max_x, b.max_x);
    box.max_y = fmaxf(a.max_y, b.max_y);
    box.max_z = fmaxf(a.max_z, b.max_z);
    return box;
}

__device__ void reduce_bounds(float mn[3], float mx[3])
{
    __shared__ float sharedMin[3][256];
    __shared__ float sharedMax[3][256];
    const int thread = threadIdx.x;
    for (int axis = 0; axis < 3; ++axis) {
        sharedMin[axis][thread] = mn[axis];
        sharedMax[axis][thread] = mx[axis];
    }
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (thread < offset) {
            for (int axis = 0; axis < 3; ++axis) {
                sharedMin[axis][thread] = fminf(sharedMin[axis][thread], sharedMin[axis][thread + offset]);
                sharedMax[axis][thread] = fmaxf(sharedMax[axis][thread], sharedMax[axis][thread + offset]);
            }
        }
        __syncthreads();
    }
    if (thread == 0) {
        for (int axis = 0; axis < 3; ++axis) {
            mn[axis] = sharedMin[axis][0];
            mx[axis] = sharedMax[axis][0];
        }
    }
}

__global__ void prepare_triangles(
    const float* vertices, const unsigned int* faces, int count,
    AABBGPU* triangleAabb, float* centroid)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    AABBGPU box;
    box.min_x = fminf(a.x, fminf(b.x, c.x));
    box.min_y = fminf(a.y, fminf(b.y, c.y));
    box.min_z = fminf(a.z, fminf(b.z, c.z));
    box.max_x = fmaxf(a.x, fmaxf(b.x, c.x));
    box.max_y = fmaxf(a.y, fmaxf(b.y, c.y));
    box.max_z = fmaxf(a.z, fmaxf(b.z, c.z));
    triangleAabb[i] = box;
    centroid[3 * i + 0] = (a.x + b.x + c.x) * (1.0f / 3.0f);
    centroid[3 * i + 1] = (a.y + b.y + c.y) * (1.0f / 3.0f);
    centroid[3 * i + 2] = (a.z + b.z + c.z) * (1.0f / 3.0f);
}

__global__ void centroid_bounds_partial(const float* centroid, int count, float* partial)
{
    float mn[3] = {1e30f, 1e30f, 1e30f};
    float mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float x = centroid[3 * i + 0];
        const float y = centroid[3 * i + 1];
        const float z = centroid[3 * i + 2];
        mn[0] = fminf(mn[0], x); mn[1] = fminf(mn[1], y); mn[2] = fminf(mn[2], z);
        mx[0] = fmaxf(mx[0], x); mx[1] = fmaxf(mx[1], y); mx[2] = fmaxf(mx[2], z);
    }
    reduce_bounds(mn, mx);
    if (threadIdx.x == 0) {
        float* out = partial + 6 * blockIdx.x;
        out[0] = mn[0]; out[1] = mn[1]; out[2] = mn[2];
        out[3] = mx[0]; out[4] = mx[1]; out[5] = mx[2];
    }
}

__global__ void centroid_bounds_final(const float* partial, int blocks, float* bounds)
{
    float mn[3] = {1e30f, 1e30f, 1e30f};
    float mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = threadIdx.x; i < blocks; i += blockDim.x) {
        const float* in = partial + 6 * i;
        mn[0] = fminf(mn[0], in[0]); mn[1] = fminf(mn[1], in[1]); mn[2] = fminf(mn[2], in[2]);
        mx[0] = fmaxf(mx[0], in[3]); mx[1] = fmaxf(mx[1], in[4]); mx[2] = fmaxf(mx[2], in[5]);
    }
    reduce_bounds(mn, mx);
    if (threadIdx.x == 0) {
        bounds[0] = mn[0]; bounds[1] = mn[1]; bounds[2] = mn[2];
        bounds[3] = mx[0]; bounds[4] = mx[1]; bounds[5] = mx[2];
    }
}

__global__ void write_morton(
    const float* centroid, int count, const float* bounds,
    unsigned int* codes, unsigned int* order)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float dx = fmaxf(bounds[3] - bounds[0], 1e-9f);
    const float dy = fmaxf(bounds[4] - bounds[1], 1e-9f);
    const float dz = fmaxf(bounds[5] - bounds[2], 1e-9f);
    float x = (centroid[3 * i + 0] - bounds[0]) / dx;
    float y = (centroid[3 * i + 1] - bounds[1]) / dy;
    float z = (centroid[3 * i + 2] - bounds[2]) / dz;
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    y = fminf(fmaxf(y, 0.0f), 1.0f);
    z = fminf(fmaxf(z, 0.0f), 1.0f);
    codes[i] = morton3D_d(x, y, z);
    order[i] = (unsigned int)i;
}

__global__ void write_leaves(
    const AABBGPU* triangleAabb, const unsigned int* order, int count,
    BVHNodeGPU* nodes, unsigned int* leafTriIndices)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const unsigned int triangle = order[i];
    const int leaf = (count - 1) + i;
    nodes[leaf].aabb = triangleAabb[triangle];
    nodes[leaf].leftChildIndex = 0xFFFFFFFFu;
    nodes[leaf].rightChildIndex = 0xFFFFFFFFu;
    leafTriIndices[i] = triangle;
}

__global__ void build_internal_nodes(
    const unsigned int* codes, int count, BVHNodeGPU* nodes, unsigned int* parent)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count - 1)
        return;
    int first, last;
    determine_range_d(codes, count, i, first, last);
    const int split = find_split_d(codes, count, first, last);
    const int left = (split == first) ? (count - 1 + split) : split;
    const int right = (split + 1 == last) ? (count - 1 + split + 1) : (split + 1);
    nodes[i].leftChildIndex = (unsigned int)left;
    nodes[i].rightChildIndex = (unsigned int)right;
    parent[left] = (unsigned int)i;
    parent[right] = (unsigned int)i;
}

__global__ void fit_internal_aabbs(
    BVHNodeGPU* nodes, const unsigned int* parent, unsigned int* visits, int count)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count)
        return;
    int idx = (count - 1) + slot;
    while (idx != 0) {
        const unsigned int above = parent[idx];
        if (atomicAdd(visits + above, 1u) == 0u)
            return;
        BVHNodeGPU& node = nodes[above];
        node.aabb = merge_aabb(nodes[node.leftChildIndex].aabb, nodes[node.rightChildIndex].aabb);
        idx = (int)above;
    }
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

void build_lbvh(
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes, gpu::gpu_mem_32u& leafTriIndices,
    gpu::gpu_mem_32u& morton, gpu::gpu_mem_32u& order,
    gpu::gpu_mem_32u& parent, gpu::gpu_mem_32u& visits,
    gpu::shared_device_buffer_typed<AABBGPU>& triangleAabb,
    gpu::gpu_mem_32f& centroids, gpu::gpu_mem_32f& partial, gpu::gpu_mem_32f& bounds,
    unsigned int nvertices, unsigned int nfaces)
{
    (void)nvertices;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124313, context.type());
    if (nfaces == 0)
        return;

    cudaStream_t stream = context.cudaStream();
    const int count = (int)nfaces;
    const int threads = 256;
    const int triangleBlocks = (count + threads - 1) / threads;
    int boundBlocks = triangleBlocks > 1024 ? 1024 : triangleBlocks;

    ::prepare_triangles<<<triangleBlocks, threads, 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), count, triangleAabb.cuptr(), centroids.cuptr());
    ::centroid_bounds_partial<<<boundBlocks, threads, 0, stream>>>(centroids.cuptr(), count, partial.cuptr());
    ::centroid_bounds_final<<<1, threads, 0, stream>>>(partial.cuptr(), boundBlocks, bounds.cuptr());
    ::write_morton<<<triangleBlocks, threads, 0, stream>>>(
        centroids.cuptr(), count, bounds.cuptr(), morton.cuptr(), order.cuptr());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), morton.cuptr(), morton.cuptr() + count, order.cuptr());
    ::write_leaves<<<triangleBlocks, threads, 0, stream>>>(
        triangleAabb.cuptr(), order.cuptr(), count, nodes.cuptr(), leafTriIndices.cuptr());
    if (count > 1) {
        cudaMemsetAsync(visits.cuptr(), 0, sizeof(unsigned int) * (unsigned int)(count - 1), stream);
        const int internalBlocks = (count - 1 + threads - 1) / threads;
        ::build_internal_nodes<<<internalBlocks, threads, 0, stream>>>(
            morton.cuptr(), count, nodes.cuptr(), parent.cuptr());
        ::fit_internal_aabbs<<<triangleBlocks, threads, 0, stream>>>(
            nodes.cuptr(), parent.cuptr(), visits.cuptr(), count);
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
