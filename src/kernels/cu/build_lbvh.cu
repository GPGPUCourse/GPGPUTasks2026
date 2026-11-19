#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/atomic>
#include <math_constants.h>

#include <limits>

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cu"

namespace {

constexpr unsigned int LBVH_BLOCK_SIZE = 256;

struct CombineBoxes {
    __host__ __device__ AABBGPU operator()(const AABBGPU &a, const AABBGPU &b) const {
        return AABBGPU{
            fminf(a.min_x, b.min_x), fminf(a.min_y, b.min_y), fminf(a.min_z, b.min_z),
            fmaxf(a.max_x, b.max_x), fmaxf(a.max_y, b.max_y), fmaxf(a.max_z, b.max_z)
        };
    }
};

__device__ void atomicMinFloat(float *address, float value) {
    unsigned int *bits = reinterpret_cast<unsigned int*>(address);
    unsigned int old = atomicCAS(bits, 0u, 0u);
    unsigned int assumed;
    do {
        assumed = old;
        const float next = fminf(__uint_as_float(assumed), value);
        old = atomicCAS(bits, assumed, __float_as_uint(next));
    } while (old != assumed);
}

__device__ void atomicMaxFloat(float *address, float value) {
    unsigned int *bits = reinterpret_cast<unsigned int*>(address);
    unsigned int old = atomicCAS(bits, 0u, 0u);
    unsigned int assumed;
    do {
        assumed = old;
        const float next = fmaxf(__uint_as_float(assumed), value);
        old = atomicCAS(bits, assumed, __float_as_uint(next));
    } while (old != assumed);
}

// Each thread writes one triangle's data, then its block combines centroids.
__global__ void lbvh_triangle_geometry(
    const float* vertices,
    const unsigned int* faces,
    unsigned int nfaces,
    float* centroids,
    AABBGPU* triangleBoxes,
    AABBGPU* centroidBounds)
{
    using BlockReduce = cub::BlockReduce<AABBGPU, LBVH_BLOCK_SIZE>;
    __shared__ BlockReduce::TempStorage tempStorage;

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const float infinity = CUDART_INF_F;
    AABBGPU pointBox{infinity, infinity, infinity, -infinity, -infinity, -infinity};

    if (index < nfaces) {
        uint3 face = loadFace(faces, index);

        float3 v0 = loadVertex(vertices, face.x);
        float3 v1 = loadVertex(vertices, face.y);
        float3 v2 = loadVertex(vertices, face.z);

        const float x = (v0.x + v1.x + v2.x) / 3;
        const float y = (v0.y + v1.y + v2.y) / 3;
        const float z = (v0.z + v1.z + v2.z) / 3;
        const unsigned int offset = 3 * index;
        centroids[offset] = x;
        centroids[offset + 1] = y;
        centroids[offset + 2] = z;

        triangleBoxes[index] = AABBGPU{min(v0.x, min(v1.x, v2.x)),
            min(v0.y, min(v1.y, v2.y)),
            min(v0.z, min(v1.z, v2.z)),
            max(v0.x, max(v1.x, v2.x)),
            max(v0.y, max(v1.y, v2.y)),
            max(v0.z, max(v1.z, v2.z))
        };
        pointBox = AABBGPU{x, y, z, x, y, z};
    }

    const AABBGPU blockBounds = BlockReduce(tempStorage).Reduce(pointBox, CombineBoxes{});
    if (threadIdx.x == 0) {
        atomicMinFloat(&centroidBounds->min_x, blockBounds.min_x);
        atomicMinFloat(&centroidBounds->min_y, blockBounds.min_y);
        atomicMinFloat(&centroidBounds->min_z, blockBounds.min_z);
        atomicMaxFloat(&centroidBounds->max_x, blockBounds.max_x);
        atomicMaxFloat(&centroidBounds->max_y, blockBounds.max_y);
        atomicMaxFloat(&centroidBounds->max_z, blockBounds.max_z);
    }
}

__device__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ unsigned int mortanCode(float x, float y, float z) {
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    return (xx << 2) | (yy << 1) | zz;
}

__global__ void lbvh_morton_codes(
    const float* centroids,
    const AABBGPU* centroidBounds,
    unsigned int nfaces,
    unsigned int* mortonCodes,
    unsigned int* triangleIndices)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nfaces)
        return;
    const unsigned int offset = 3 * index;
    float x = (centroids[offset] - centroidBounds->min_x) / max(1e-9f, centroidBounds->max_x - centroidBounds->min_x);
    float y = (centroids[offset + 1] - centroidBounds->min_y) / max(1e-9f, centroidBounds->max_y - centroidBounds->min_y);
    float z = (centroids[offset + 2] - centroidBounds->min_z) / max(1e-9f, centroidBounds->max_z - centroidBounds->min_z);
    mortonCodes[index] = mortanCode(x, y, z);
    triangleIndices[index] = index;
}

__device__ int commonPrefix(const unsigned int* codes, unsigned int nfaces,
                            unsigned int index, long long other)
{
    if (other < 0 || other >= nfaces)
        return -1;

    const unsigned int otherIndex = (unsigned int) other;
    const unsigned int codeDiff = codes[index] ^ codes[otherIndex];
    if (codeDiff != 0)
        return __clz(codeDiff);

    return 32 + __clz(index ^ otherIndex);
}

__device__ void determineRange(const unsigned int* codes, unsigned int nfaces,
                               unsigned int index, unsigned int &first, unsigned int &last)
{
    const long long i = index;
    const int prefixLeft = commonPrefix(codes, nfaces, index, i - 1);
    const int prefixRight = commonPrefix(codes, nfaces, index, i + 1);
    const int direction = prefixRight > prefixLeft ? 1 : -1;
    const int minPrefix = commonPrefix(codes, nfaces, index, i - direction);

    long long maxLength = 2;
    while (commonPrefix(codes, nfaces, index, i + maxLength * direction) > minPrefix)
        maxLength *= 2;

    long long length = 0;
    for (long long step = maxLength / 2; step > 0; step /= 2)
        if (commonPrefix(codes, nfaces, index, i + (length + step) * direction) > minPrefix)
            length += step;

    const unsigned int other = (unsigned int) (i + length * direction);
    first = min(index, other);
    last = max(index, other);
}

__device__ unsigned int findSplit(const unsigned int* codes, unsigned int nfaces,
                                  unsigned int first, unsigned int last)
{
    const int rangePrefix = commonPrefix(codes, nfaces, first, last);
    unsigned int split = first;
    unsigned int step = last - first;

    do {
        step = (step + 1) / 2;
        const unsigned int candidate = split + step;
        if (candidate < last && commonPrefix(codes, nfaces, first, candidate) > rangePrefix)
            split = candidate;
    } while (step > 1);

    return split;
}

// Each thread builds one internal node. Leaf data and boxes are filled later.
__global__ void lbvh_build_internal_nodes(
    const unsigned int* sortedCodes,
    unsigned int nfaces,
    BVHNodeGPU* nodes,
    unsigned int* parents)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (nfaces < 2 || index >= nfaces - 1)
        return;

    unsigned int first, last;
    determineRange(sortedCodes, nfaces, index, first, last);
    const unsigned int split = findSplit(sortedCodes, nfaces, first, last);

    const unsigned int left = split == first ? nfaces - 1 + split : split;
    const unsigned int right = split + 1 == last ? nfaces + split : split + 1;
    nodes[index].leftChildIndex = left;
    nodes[index].rightChildIndex = right;
    parents[left] = index;
    parents[right] = index;
    if (index == 0)
        parents[0] = 0xffffffffu;
}

// leaves bbox initialization
__global__ void lbvh_initialize_leaves(
    const unsigned int* sortedIndices,
    const AABBGPU* triangleBoxes,
    unsigned int nfaces,
    BVHNodeGPU* nodes,
    unsigned int* parents)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nfaces)
        return;

    if (nfaces == 1)
        parents[index] = 0xffffffffu;

    const unsigned int leaf_index = index + nfaces - 1;

    nodes[leaf_index].leftChildIndex = 0xffffffffu;
    nodes[leaf_index].rightChildIndex = 0xffffffffu;
    nodes[leaf_index].aabb = triangleBoxes[sortedIndices[index]];
}

// build bboxes for all nodes
__global__ void lbvh_build_boxes(
    BVHNodeGPU* nodes,
    const unsigned int* parents,
    unsigned int* arrivals,
    unsigned int nfaces)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nfaces)
        return;

    unsigned int node = nfaces - 1 + index;
    while (parents[node] != 0xffffffffu) {
        const unsigned int parent = parents[node];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> arrived(arrivals[parent]);
        const unsigned int previous = arrived.fetch_add(1, cuda::memory_order_acq_rel);

        if (previous == 0)
            return;

        const unsigned int left = nodes[parent].leftChildIndex;
        const unsigned int right = nodes[parent].rightChildIndex;
        nodes[parent].aabb = CombineBoxes{}(nodes[left].aabb, nodes[right].aabb);
        node = parent;
    }
}

} // namespace

namespace cuda {

void lbvh_morton_codes(
    const gpu::gpu_mem_32f &centroids,
    const gpu::shared_device_buffer_typed<AABBGPU> &centroidBounds,
    unsigned int nfaces,
    gpu::gpu_mem_32u &mortonCodes,
    gpu::gpu_mem_32u &triangleIndices)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 835175866, context.type());
    rassert(nfaces > 0, 835175867);
    rassert(centroids.number() >= (size_t) nfaces * 3, 835175868);
    rassert(centroidBounds.number() >= 1, 835175869);
    rassert(mortonCodes.number() >= nfaces && triangleIndices.number() >= nfaces, 835175870);

    const unsigned int blocks = nfaces / LBVH_BLOCK_SIZE + (nfaces % LBVH_BLOCK_SIZE != 0);
    cudaStream_t stream = context.cudaStream();
    ::lbvh_morton_codes<<<blocks, LBVH_BLOCK_SIZE, 0, stream>>>(
        centroids.cuptr(), centroidBounds.cuptr(), nfaces, mortonCodes.cuptr(), triangleIndices.cuptr());
    CUDA_CHECK_KERNEL(stream);
}

void lbvh_initialize_leaves(
    const gpu::gpu_mem_32u &sortedIndices,
    const gpu::shared_device_buffer_typed<AABBGPU> &triangleBoxes,
    unsigned int nfaces,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes,
    gpu::gpu_mem_32u &parents)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 835175871, context.type());
    rassert(nfaces > 0 && nfaces <= 0x80000000u, 835175872);
    const size_t nodeCount = (size_t) nfaces * 2 - 1;
    rassert(sortedIndices.number() >= nfaces && triangleBoxes.number() >= nfaces, 835175873);
    rassert(nodes.number() >= nodeCount && parents.number() >= nodeCount, 835175874);

    const unsigned int blocks = nfaces / LBVH_BLOCK_SIZE + (nfaces % LBVH_BLOCK_SIZE != 0);
    cudaStream_t stream = context.cudaStream();
    ::lbvh_initialize_leaves<<<blocks, LBVH_BLOCK_SIZE, 0, stream>>>(
        sortedIndices.cuptr(), triangleBoxes.cuptr(), nfaces, nodes.cuptr(), parents.cuptr());
    CUDA_CHECK_KERNEL(stream);
}

void lbvh_build_boxes(
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes,
    const gpu::gpu_mem_32u &parents,
    gpu::gpu_mem_32u &arrivals,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 835175858, context.type());
    rassert(nfaces > 0 && nfaces <= 0x80000000u, 835175859);
    const size_t nodeCount = (size_t) nfaces * 2 - 1;
    rassert(nodes.number() >= nodeCount, 835175860);
    rassert(parents.number() >= nodeCount, 835175861);
    rassert(arrivals.number() >= nfaces - 1, 835175862);

    if (nfaces == 1)
        return;

    cudaStream_t stream = context.cudaStream();
    CUDA_SAFE_CALL(cudaMemsetAsync(arrivals.cuptr(), 0, (size_t) (nfaces - 1) * sizeof(unsigned int), stream));
    const unsigned int blocks = nfaces / LBVH_BLOCK_SIZE + (nfaces % LBVH_BLOCK_SIZE != 0);
    ::lbvh_build_boxes<<<blocks, LBVH_BLOCK_SIZE, 0, stream>>>(
        nodes.cuptr(), parents.cuptr(), arrivals.cuptr(), nfaces);
    CUDA_CHECK_KERNEL(stream);
}

void lbvh_build_internal_nodes(
    const gpu::gpu_mem_32u &sortedCodes,
    unsigned int nfaces,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes,
    gpu::gpu_mem_32u &parents)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 835175852, context.type());
    rassert(nfaces >= 2 && nfaces <= 0x80000000u, 835175853);
    const size_t nodeCount = (size_t) nfaces * 2 - 1;
    rassert(sortedCodes.number() >= nfaces, 835175854);
    rassert(nodes.number() >= nodeCount, 835175855);
    rassert(parents.number() >= nodeCount, 835175856);

    const unsigned int internalNodes = nfaces - 1;
    const unsigned int blocks = internalNodes / LBVH_BLOCK_SIZE + (internalNodes % LBVH_BLOCK_SIZE != 0);
    cudaStream_t stream = context.cudaStream();
    ::lbvh_build_internal_nodes<<<blocks, LBVH_BLOCK_SIZE, 0, stream>>>(
        sortedCodes.cuptr(), nfaces, nodes.cuptr(), parents.cuptr());
    CUDA_CHECK_KERNEL(stream);
}

void lbvh_sort_morton_codes(
    const gpu::gpu_mem_32u &mortonCodes,
    const gpu::gpu_mem_32u &triangleIndices,
    unsigned int nfaces,
    gpu::gpu_mem_32u &sortedCodes,
    gpu::gpu_mem_32u &sortedIndices,
    gpu::shared_device_buffer &workspace)
{
    size_t workspaceBytes = 0;
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();

    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
        nullptr, workspaceBytes,
        mortonCodes.cuptr(), sortedCodes.cuptr(),
        triangleIndices.cuptr(), sortedIndices.cuptr(),
        nfaces, 0, 30, stream
    ));

    workspace.grow(workspaceBytes);

    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortPairs(
        workspace.cuptr(), workspaceBytes,
        mortonCodes.cuptr(), sortedCodes.cuptr(),
        triangleIndices.cuptr(), sortedIndices.cuptr(),
        nfaces, 0, 30, stream
    ));
}

void lbvh_triangle_geometry(
    const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u &faces,
    unsigned int nfaces,
    gpu::gpu_mem_32f &centroids,
    gpu::shared_device_buffer_typed<AABBGPU> &triangleBoxes,
    gpu::shared_device_buffer_typed<AABBGPU> &centroidBounds)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 835175843, context.type());
    rassert(nfaces > 0, 835175844);
    rassert(faces.number() >= static_cast<size_t>(nfaces) * 3, 835175848);
    rassert(centroids.number() >= static_cast<size_t>(nfaces) * 3, 835175845);
    rassert(triangleBoxes.number() >= nfaces, 835175849);
    rassert(centroidBounds.number() >= 1, 835175846);

    const float infinity = std::numeric_limits<float>::infinity();
    const AABBGPU emptyBox{infinity, infinity, infinity, -infinity, -infinity, -infinity};
    centroidBounds.writeN(&emptyBox, 1);

    const cudaStream_t stream = context.cudaStream();
    const unsigned int blocks = nfaces / LBVH_BLOCK_SIZE + (nfaces % LBVH_BLOCK_SIZE != 0);
    ::lbvh_triangle_geometry<<<blocks, LBVH_BLOCK_SIZE, 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), nfaces, centroids.cuptr(),
        triangleBoxes.cuptr(), centroidBounds.cuptr());
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
