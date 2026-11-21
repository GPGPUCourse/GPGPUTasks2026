#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include <cub/device/device_radix_sort.cuh>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "geometry_helpers.cu"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
static __device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
static __device__ unsigned int morton3D(float x, float y, float z)
{
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);
    return (expandBits(ix) << 2) | (expandBits(iy) << 1) | expandBits(iz);
}

// Монотонное отображение float -> uint, чтобы min/max считались целочисленными атомиками
__device__ unsigned int float_flip(float f)
{
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ float float_unflip(unsigned int u)
{
    u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return __uint_as_float(u);
}

__device__ float3 centroid(const float* vertices, const unsigned int* faces, unsigned int fi)
{
    uint3 f = loadFace(faces, fi);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);
    return make_float3((a.x + b.x + c.x) * (1.0f / 3.0f),
                       (a.y + b.y + c.y) * (1.0f / 3.0f),
                       (a.z + b.z + c.z) * (1.0f / 3.0f));
}

// Compute common prefix length between sorted Morton codes at indices i and j
// Uses index as a tiebreaker when codes are equal (like Karras 2013).
__device__ int common_prefix(const unsigned int* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;
    unsigned int ci = codes[i];
    unsigned int cj = codes[j];
    if (ci == cj) {
        return 32 + __clz((unsigned int) i ^ (unsigned int) j);
    } else {
        return __clz(ci ^ cj);
    }
}

// Determine range [first, last] of primitives covered by internal node i
__device__ void determine_range(const unsigned int* codes, int N, int i, int& outFirst, int& outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    int d = (cpR > cpL) ? 1 : -1;

    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;
    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    outFirst = min(i, j);
    outLast = max(i, j);
}

// Find split position inside range [first, last]
__device__ int find_split(const unsigned int* codes, int N, int first, int last)
{
    if (first == last)
        return first;

    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step = last - first;
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last && common_prefix(codes, N, first, newSplit) > commonPrefix) {
            split = newSplit;
        }
    } while (step > 1);

    return split;
}

// bounds[0..2] - min центроидов, bounds[3..5] - max (в представлении float_flip)
__global__ void lbvh_01_centroid_bounds(const float* vertices, const unsigned int* faces,
                                        unsigned int* bounds, unsigned int nfaces)
{
    __shared__ unsigned int s_min[3];
    __shared__ unsigned int s_max[3];
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < 3) {
        s_min[threadIdx.x] = 0xFFFFFFFFu;
        s_max[threadIdx.x] = 0u;
    }
    __syncthreads();

    if (i < nfaces) {
        float3 c = centroid(vertices, faces, i);
        unsigned int cx = float_flip(c.x), cy = float_flip(c.y), cz = float_flip(c.z);
        atomicMin(&s_min[0], cx); atomicMax(&s_max[0], cx);
        atomicMin(&s_min[1], cy); atomicMax(&s_max[1], cy);
        atomicMin(&s_min[2], cz); atomicMax(&s_max[2], cz);
    }
    __syncthreads();

    if (threadIdx.x < 3) {
        atomicMin(&bounds[threadIdx.x], s_min[threadIdx.x]);
        atomicMax(&bounds[3 + threadIdx.x], s_max[threadIdx.x]);
    }
}

__global__ void lbvh_02_morton_codes(const float* vertices, const unsigned int* faces,
                                     const unsigned int* bounds,
                                     unsigned int* codes, unsigned int* indices, unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    const float eps = 1e-9f;
    float3 cmin = make_float3(float_unflip(bounds[0]), float_unflip(bounds[1]), float_unflip(bounds[2]));
    float3 cmax = make_float3(float_unflip(bounds[3]), float_unflip(bounds[4]), float_unflip(bounds[5]));
    float3 c = centroid(vertices, faces, i);
    float nx = (c.x - cmin.x) / fmaxf(cmax.x - cmin.x, eps);
    float ny = (c.y - cmin.y) / fmaxf(cmax.y - cmin.y, eps);
    float nz = (c.z - cmin.z) / fmaxf(cmax.z - cmin.z, eps);
    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f);

    codes[i] = morton3D(nx, ny, nz);
    indices[i] = i;
}

// Листья [N-1 .. 2N-2] с AABB треугольников, внутренние узлы [0 .. N-2] с детьми (Karras 2012)
__global__ void lbvh_03_nodes(const float* vertices, const unsigned int* faces,
                              const unsigned int* sorted_codes, const unsigned int* sorted_indices,
                              BVHNodeGPU* nodes, unsigned int* parents, unsigned int* visited, unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    const int N = (int) nfaces;
    const unsigned int INVALID = 0xFFFFFFFFu;

    {
        uint3 f = loadFace(faces, sorted_indices[i]);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);
        BVHNodeGPU leaf;
        leaf.aabb.min_x = fminf(fminf(a.x, b.x), c.x);
        leaf.aabb.min_y = fminf(fminf(a.y, b.y), c.y);
        leaf.aabb.min_z = fminf(fminf(a.z, b.z), c.z);
        leaf.aabb.max_x = fmaxf(fmaxf(a.x, b.x), c.x);
        leaf.aabb.max_y = fmaxf(fmaxf(a.y, b.y), c.y);
        leaf.aabb.max_z = fmaxf(fmaxf(a.z, b.z), c.z);
        leaf.leftChildIndex = INVALID;
        leaf.rightChildIndex = INVALID;
        nodes[(N - 1) + i] = leaf;
    }

    if (i >= nfaces - 1)
        return;

    visited[i] = 0;
    if (i == 0)
        parents[0] = INVALID;

    int first, last;
    determine_range(sorted_codes, N, (int) i, first, last);
    int split = find_split(sorted_codes, N, first, last);

    unsigned int left = (split == first) ? (unsigned int) ((N - 1) + split) : (unsigned int) split;
    unsigned int right = (split + 1 == last) ? (unsigned int) ((N - 1) + split + 1) : (unsigned int) (split + 1);

    nodes[i].leftChildIndex = left;
    nodes[i].rightChildIndex = right;
    parents[left] = i;
    parents[right] = i;
}

__device__ AABBGPU load_aabb_cg(const BVHNodeGPU* nodes, unsigned int c)
{
    // L1 не когерентен между SM, AABB детей только что записан другим блоком - читаем мимо L1
    AABBGPU box;
    box.min_x = __ldcg(&nodes[c].aabb.min_x);
    box.min_y = __ldcg(&nodes[c].aabb.min_y);
    box.min_z = __ldcg(&nodes[c].aabb.min_z);
    box.max_x = __ldcg(&nodes[c].aabb.max_x);
    box.max_y = __ldcg(&nodes[c].aabb.max_y);
    box.max_z = __ldcg(&nodes[c].aabb.max_z);
    return box;
}

// Снизу вверх от листьев: второй пришедший в узел ребёнок объединяет AABB и идёт к родителю
__global__ void lbvh_04_aabbs(BVHNodeGPU* nodes, const unsigned int* parents, unsigned int* visited, unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    unsigned int node = (nfaces - 1) + i;
    while (true) {
        unsigned int p = parents[node];
        if (p == 0xFFFFFFFFu)
            return;
        __threadfence();
        if (atomicAdd(&visited[p], 1) == 0)
            return;

        AABBGPU l = load_aabb_cg(nodes, nodes[p].leftChildIndex);
        AABBGPU r = load_aabb_cg(nodes, nodes[p].rightChildIndex);
        AABBGPU box;
        box.min_x = fminf(l.min_x, r.min_x);
        box.min_y = fminf(l.min_y, r.min_y);
        box.min_z = fminf(l.min_z, r.min_z);
        box.max_x = fmaxf(l.max_x, r.max_x);
        box.max_y = fmaxf(l.max_y, r.max_y);
        box.max_z = fmaxf(l.max_z, r.max_z);
        nodes[p].aabb = box;
        node = p;
    }
}

namespace cuda {
size_t build_lbvh_temp_bytes(unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    size_t temp_bytes = 0;
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        (const unsigned int*) nullptr, (unsigned int*) nullptr,
        (const unsigned int*) nullptr, (unsigned int*) nullptr,
        (int) nfaces, 0, 30, context.cudaStream()));
    return temp_bytes;
}

void build_lbvh(const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces, unsigned int nfaces,
    gpu::gpu_mem_32u &centroid_bounds, gpu::gpu_mem_32u &morton_codes, gpu::gpu_mem_32u &faces_indices,
    gpu::gpu_mem_32u &sorted_morton_codes, gpu::gpu_mem_32u &leaf_faces_indices, gpu::gpu_mem_8u &sort_temp,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &parents, gpu::gpu_mem_32u &visited)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize ws(GROUP_SIZE, nfaces);

    CUDA_SAFE_CALL(cudaMemsetAsync(centroid_bounds.cuptr(), 0xFF, 3 * sizeof(unsigned int), stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(centroid_bounds.cuptr() + 3, 0x00, 3 * sizeof(unsigned int), stream));
    ::lbvh_01_centroid_bounds<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), centroid_bounds.cuptr(), nfaces);
    CUDA_CHECK_KERNEL(stream);

    ::lbvh_02_morton_codes<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), centroid_bounds.cuptr(), morton_codes.cuptr(), faces_indices.cuptr(), nfaces);
    CUDA_CHECK_KERNEL(stream);

    size_t temp_bytes = sort_temp.size();
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(sort_temp.cuptr(), temp_bytes,
        morton_codes.cuptr(), sorted_morton_codes.cuptr(),
        faces_indices.cuptr(), leaf_faces_indices.cuptr(),
        (int) nfaces, 0, 30, stream));
    CUDA_CHECK_KERNEL(stream);

    ::lbvh_03_nodes<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), sorted_morton_codes.cuptr(), leaf_faces_indices.cuptr(),
        nodes.cuptr(), parents.cuptr(), visited.cuptr(), nfaces);
    CUDA_CHECK_KERNEL(stream);

    ::lbvh_04_aabbs<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>(
        nodes.cuptr(), parents.cuptr(), visited.cuptr(), nfaces);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
