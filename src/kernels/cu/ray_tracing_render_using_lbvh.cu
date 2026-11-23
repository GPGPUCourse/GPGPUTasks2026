#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

namespace {
__device__ __forceinline__ unsigned int expand_morton_bits(unsigned int value)
{
    value = (value * 0x00010001u) & 0xFF0000FFu;
    value = (value * 0x00000101u) & 0x0F00F00Fu;
    value = (value * 0x00000011u) & 0xC30C30C3u;
    value = (value * 0x00000005u) & 0x49249249u;
    return value;
}

__device__ __forceinline__ unsigned int morton_code_3d(float x, float y, float z)
{
    const unsigned int ix = min(max(static_cast<int>(x * 1024.0f), 0), 1023);
    const unsigned int iy = min(max(static_cast<int>(y * 1024.0f), 0), 1023);
    const unsigned int iz = min(max(static_cast<int>(z * 1024.0f), 0), 1023);
    return (expand_morton_bits(ix) << 2) |
           (expand_morton_bits(iy) << 1) |
            expand_morton_bits(iz);
}

__device__ __forceinline__ int common_prefix_length(
    const unsigned int* codes, int count, int i, int j)
{
    if (j < 0 || j >= count)
        return -1;

    const unsigned int a = codes[i];
    const unsigned int b = codes[j];
    if (a != b)
        return __clz(a ^ b);

    return 32 + __clz(static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j));
}

__device__ void determine_leaf_range(
    const unsigned int* codes, int count, int i, int& first, int& last)
{
    const int prefix_left = common_prefix_length(codes, count, i, i - 1);
    const int prefix_right = common_prefix_length(codes, count, i, i + 1);
    const int direction = (prefix_right > prefix_left) ? 1 : -1;
    const int prefix_min = common_prefix_length(codes, count, i, i - direction);

    int max_length = 2;
    while (common_prefix_length(codes, count, i, i + max_length * direction) > prefix_min)
        max_length <<= 1;

    int length = 0;
    for (int step = max_length >> 1; step > 0; step >>= 1) {
        if (common_prefix_length(codes, count, i, i + (length + step) * direction) > prefix_min)
            length += step;
    }

    const int j = i + length * direction;
    first = min(i, j);
    last = max(i, j);
}

__device__ int find_leaf_split(const unsigned int* codes, int count, int first, int last)
{
    if (first == last)
        return first;

    const int common = common_prefix_length(codes, count, first, last);
    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        const int candidate = split + step;
        if (candidate < last && common_prefix_length(codes, count, first, candidate) > common)
            split = candidate;
    } while (step > 1);
    return split;
}

__device__ __forceinline__ AABBGPU union_aabb(const AABBGPU& a, const AABBGPU& b)
{
    AABBGPU result;
    result.min_x = min(a.min_x, b.min_x);
    result.min_y = min(a.min_y, b.min_y);
    result.min_z = min(a.min_z, b.min_z);
    result.max_x = max(a.max_x, b.max_x);
    result.max_y = max(a.max_y, b.max_y);
    result.max_z = max(a.max_z, b.max_z);
    return result;
}
} // namespace

__global__ void compute_morton_codes(
    const float* vertices, const unsigned int* faces,
    unsigned int* codes, unsigned int* leaf_tri_indices,
    unsigned int nfaces,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x);
    const float3 b = loadVertex(vertices, face.y);
    const float3 c = loadVertex(vertices, face.z);
    const float cx = (a.x + b.x + c.x) * (1.0f / 3.0f);
    const float cy = (a.y + b.y + c.y) * (1.0f / 3.0f);
    const float cz = (a.z + b.z + c.z) * (1.0f / 3.0f);

    const float dx = fmaxf(max_x - min_x, 1e-9f);
    const float dy = fmaxf(max_y - min_y, 1e-9f);
    const float dz = fmaxf(max_z - min_z, 1e-9f);
    const float nx = fminf(fmaxf((cx - min_x) / dx, 0.0f), 1.0f);
    const float ny = fminf(fmaxf((cy - min_y) / dy, 0.0f), 1.0f);
    const float nz = fminf(fmaxf((cz - min_z) / dz, 0.0f), 1.0f);

    codes[i] = morton_code_3d(nx, ny, nz);
    leaf_tri_indices[i] = i;
}

__global__ void initialize_lbvh_leaves(
    const float* vertices, const unsigned int* faces,
    const unsigned int* leaf_tri_indices, BVHNodeGPU* nodes,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    const unsigned int face_index = leaf_tri_indices[i];
    const uint3 face = loadFace(faces, face_index);
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

    const unsigned int leaf_index = nfaces - 1 + i;
    nodes[leaf_index].aabb = box;
    nodes[leaf_index].leftChildIndex = 0xffffffffu;
    nodes[leaf_index].rightChildIndex = 0xffffffffu;
}

__global__ void construct_lbvh_internal_nodes(
    const unsigned int* codes, BVHNodeGPU* nodes,
    unsigned int* parents, unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces - 1)
        return;

    const int count = static_cast<int>(nfaces);
    int first, last;
    determine_leaf_range(codes, count, static_cast<int>(i), first, last);
    const int split = find_leaf_split(codes, count, first, last);

    const unsigned int left = (split == first)
        ? nfaces - 1 + static_cast<unsigned int>(split)
        : static_cast<unsigned int>(split);
    const unsigned int right = (split + 1 == last)
        ? nfaces - 1 + static_cast<unsigned int>(split + 1)
        : static_cast<unsigned int>(split + 1);

    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    parents[left] = i;
    parents[right] = i;
}

__global__ void propagate_lbvh_bounds(
    BVHNodeGPU* nodes, const unsigned int* parents,
    unsigned int* arrival_counts, unsigned int nfaces)
{
    const unsigned int leaf_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_id >= nfaces)
        return;

    unsigned int node = nfaces - 1 + leaf_id;
    unsigned int parent = parents[node];
    while (parent != 0xffffffffu) {
        __threadfence();
        const unsigned int old_arrivals = atomicAdd(&arrival_counts[parent], 1u);
        if (old_arrivals == 0u)
            return;
        if (old_arrivals != 1u)
            return;

        const unsigned int left = nodes[parent].leftChildIndex;
        const unsigned int right = nodes[parent].rightChildIndex;
        nodes[parent].aabb = union_aabb(nodes[left].aabb, nodes[right].aabb);
        node = parent;
        parent = parents[node];
    }
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
    constexpr int STACK_CAPACITY = 128;
    unsigned int stack[STACK_CAPACITY];
    int stack_size = 0;
    stack[stack_size++] = rootIndex;

    while (stack_size > 0) {
        const unsigned int node_index = stack[--stack_size];
        float box_near, box_far;
        if (!intersect_ray_aabb(orig, dir, nodes[node_index].aabb,
                                tMin, outT, box_near, box_far))
            continue;

        if (node_index >= static_cast<unsigned int>(leafStart)) {
            const unsigned int sorted_leaf = node_index - leafStart;
            const unsigned int face_index = leafTriIndices[sorted_leaf];
            const uint3 face = loadFace(faces, face_index);
            const float3 v0 = loadVertex(vertices, face.x);
            const float3 v1 = loadVertex(vertices, face.y);
            const float3 v2 = loadVertex(vertices, face.z);
            float t, u, v;
            if (intersect_ray_triangle(orig, dir, v0, v1, v2,
                                       tMin, outT, false, t, u, v)) {
                outT = t;
                outFaceId = static_cast<int>(face_index);
                outU = u;
                outV = v;
            }
            continue;
        }

        if (stack_size + 2 > STACK_CAPACITY) {
            // Safety fallback: keep correctness even for an unexpectedly deep tree.
            for (unsigned int face_index = 0; face_index < nfaces; ++face_index) {
                const uint3 face = loadFace(faces, face_index);
                const float3 v0 = loadVertex(vertices, face.x);
                const float3 v1 = loadVertex(vertices, face.y);
                const float3 v2 = loadVertex(vertices, face.z);
                float t, u, v;
                if (intersect_ray_triangle(orig, dir, v0, v1, v2,
                                           tMin, outT, false, t, u, v)) {
                    outT = t;
                    outFaceId = static_cast<int>(face_index);
                    outU = u;
                    outV = v;
                }
            }
            return outFaceId >= 0;
        }

        // Visit the left subtree first; the closest-hit bound prunes later nodes.
        stack[stack_size++] = nodes[node_index].rightChildIndex;
        stack[stack_size++] = nodes[node_index].leftChildIndex;
    }

    return outFaceId >= 0;
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
    constexpr int STACK_CAPACITY = 128;
    unsigned int stack[STACK_CAPACITY];
    int stack_size = 0;
    stack[stack_size++] = rootIndex;
    const float tMin = 1e-4f;

    while (stack_size > 0) {
        const unsigned int node_index = stack[--stack_size];
        float box_near, box_far;
        if (!intersect_ray_aabb(orig, dir, nodes[node_index].aabb,
                                tMin, FLT_MAX, box_near, box_far))
            continue;

        if (node_index >= static_cast<unsigned int>(leafStart)) {
            const unsigned int sorted_leaf = node_index - leafStart;
            const unsigned int face_index = leafTriIndices[sorted_leaf];
            if (static_cast<int>(face_index) == ignore_face)
                continue;
            const uint3 face = loadFace(faces, face_index);
            const float3 v0 = loadVertex(vertices, face.x);
            const float3 v1 = loadVertex(vertices, face.y);
            const float3 v2 = loadVertex(vertices, face.z);
            float t, u, v;
            if (intersect_ray_triangle(orig, dir, v0, v1, v2,
                                       tMin, FLT_MAX, false, t, u, v))
                return true;
            continue;
        }

        if (stack_size + 2 > STACK_CAPACITY) {
            for (unsigned int face_index = 0; face_index < nfaces; ++face_index) {
                if (static_cast<int>(face_index) == ignore_face)
                    continue;
                const uint3 face = loadFace(faces, face_index);
                const float3 v0 = loadVertex(vertices, face.x);
                const float3 v1 = loadVertex(vertices, face.y);
                const float3 v2 = loadVertex(vertices, face.z);
                float t, u, v;
                if (intersect_ray_triangle(orig, dir, v0, v1, v2,
                                           tMin, FLT_MAX, false, t, u, v))
                    return true;
            }
            return false;
        }

        stack[stack_size++] = nodes[node_index].rightChildIndex;
        stack[stack_size++] = nodes[node_index].leftChildIndex;
    }

    return false;
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

    curassert(camera->magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 946435342);
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
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes,
    gpu::gpu_mem_32u &leafTriIndices, unsigned int nfaces,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 7654321012345, context.type());
    rassert(nfaces > 0, 7654321012346);
    rassert(nodes.number() == 2ull * nfaces - 1ull, 7654321012347, nodes.number(), nfaces);
    rassert(leafTriIndices.number() == nfaces, 7654321012348, leafTriIndices.number(), nfaces);

    cudaStream_t stream = context.cudaStream();
    gpu::gpu_mem_32u mortonCodes(nfaces);
    const size_t node_count = 2ull * nfaces - 1ull;
    gpu::gpu_mem_32u parents(node_count);
    gpu::gpu_mem_32u arrivalCounts(nfaces);

    CUDA_SAFE_CALL(cudaMemsetAsync(parents.cuptr(), 0xff,
                                   node_count * sizeof(unsigned int), stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(arrivalCounts.cuptr(), 0,
                                   nfaces * sizeof(unsigned int), stream));

    ::compute_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), mortonCodes.cuptr(), leafTriIndices.cuptr(), nfaces,
        minX, minY, minZ, maxX, maxY, maxZ);
    CUDA_CHECK_KERNEL(stream);

    auto code_begin = thrust::device_pointer_cast(mortonCodes.cuptr());
    auto leaf_begin = thrust::device_pointer_cast(leafTriIndices.cuptr());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                               code_begin, code_begin + nfaces, leaf_begin);

    ::initialize_lbvh_leaves<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), leafTriIndices.cuptr(), nodes.cuptr(), nfaces);
    CUDA_CHECK_KERNEL(stream);

    if (nfaces > 1) {
        ::construct_lbvh_internal_nodes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            mortonCodes.cuptr(), nodes.cuptr(), parents.cuptr(), nfaces);
        CUDA_CHECK_KERNEL(stream);

        ::propagate_lbvh_bounds<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            nodes.cuptr(), parents.cuptr(), arrivalCounts.cuptr(), nfaces);
        CUDA_CHECK_KERNEL(stream);
    }

    CUDA_CHECK_KERNEL_SYNC(stream);
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
