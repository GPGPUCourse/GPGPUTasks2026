#include "../defines.h"
#include "../kernels.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cu"

#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

__global__ void fill_lbvh_leaves(
    const AABBGPU*, const unsigned int*, BVHNodeGPU*, int);

__global__ void build_lbvh_bounds(
    BVHNodeGPU*, const unsigned int*, unsigned int*, int);

__device__ int common_prefix(
    const unsigned int* codes,
    int n,
    int i,
    int j)
{
    if (j < 0 || j >= n)
        return -1;

    unsigned int a = codes[i];
    unsigned int b = codes[j];

    if (a == b)
        return 32 + __clz((unsigned int)i ^ (unsigned int)j);

    return __clz(a ^ b);
}

__device__ void determine_range(
    const unsigned int* codes,
    int n,
    int i,
    int& first,
    int& last)
{
    int leftPrefix = common_prefix(codes, n, i, i - 1);
    int rightPrefix = common_prefix(codes, n, i, i + 1);
    int direction = rightPrefix > leftPrefix ? 1 : -1;

    int minPrefix = common_prefix(codes, n, i, i - direction);

    int maxLength = 2;
    while (common_prefix(codes, n, i, i + maxLength * direction) > minPrefix)
        maxLength *= 2;

    int length = 0;
    for (int step = maxLength / 2; step > 0; step /= 2) {
        int nextLength = length + step;
        int j = i + nextLength * direction;

        if (common_prefix(codes, n, i, j) > minPrefix)
            length = nextLength;
    }

    int j = i + length * direction;
    first = i < j ? i : j;
    last = i > j ? i : j;
}

__device__ int find_split(
    const unsigned int* codes,
    int n,
    int first,
    int last)
{
    int prefix = common_prefix(codes, n, first, last);

    int left = first;
    int right = last;

    while (right - left > 1) {
        int middle = left + (right - left) / 2;

        if (common_prefix(codes, n, first, middle) > prefix)
            left = middle;
        else
            right = middle;
    }

    return left;
}

__global__ void build_lbvh_topology(
    const unsigned int* codes,
    BVHNodeGPU* nodes,
    unsigned int* parents,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1)
        return;

    int first, last;
    determine_range(codes, n, i, first, last);

    int split = find_split(codes, n, first, last);

    int leafStart = n - 1;

    int leftChild = split;
    if (first == split)
        leftChild = leafStart + split;

    int rightChild = split + 1;
    if (split + 1 == last)
        rightChild = leafStart + split + 1;

    nodes[i].leftChildIndex = leftChild;
    nodes[i].rightChildIndex = rightChild;

    parents[leftChild] = i;
    parents[rightChild] = i;
}

__global__ void compute_triangle_data(
    const float* vertices,
    const unsigned int* faces,
    AABBGPU* triangleBoxes,
    float* centroids,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint3 face = loadFace(faces, i);
    float3 a = loadVertex(vertices, face.x);
    float3 b = loadVertex(vertices, face.y);
    float3 c = loadVertex(vertices, face.z);

    AABBGPU& box = triangleBoxes[i];

    box.min_x = fminf(a.x, fminf(b.x, c.x));
    box.min_y = fminf(a.y, fminf(b.y, c.y));
    box.min_z = fminf(a.z, fminf(b.z, c.z));

    box.max_x = fmaxf(a.x, fmaxf(b.x, c.x));
    box.max_y = fmaxf(a.y, fmaxf(b.y, c.y));
    box.max_z = fmaxf(a.z, fmaxf(b.z, c.z));

    centroids[3 * i] = (a.x + b.x + c.x) / 3.0f;
    centroids[3 * i + 1] = (a.y + b.y + c.y) / 3.0f;
    centroids[3 * i + 2] = (a.z + b.z + c.z) / 3.0f;
}

namespace cuda {

void build_lbvh(
    const gpu::gpu_mem_32f& vertices,
    const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes,
    gpu::gpu_mem_32u& triangleIndices,
    int n)
{
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();

    nodes.resizeN(size_t(n) * 2 - 1);
    triangleIndices.resizeN(n);

    gpu::shared_device_buffer_typed<AABBGPU> triangleBoxes(n);
    gpu::gpu_mem_32f centroids(size_t(n) * 3);
    gpu::gpu_mem_32u codes(n);
    gpu::gpu_mem_32u parents(size_t(n) * 2 - 1);
    gpu::gpu_mem_32u ready;

    gpu::WorkSize workSize(GROUP_SIZE, n);

    ::compute_triangle_data<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        triangleBoxes.cuptr(), centroids.cuptr(), n);

    auto bounds = centroid_bounds(centroids, n);

    prepare_morton_codes(
        workSize, centroids, bounds, codes, triangleIndices, n);

    ::fill_lbvh_leaves<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        triangleBoxes.cuptr(), triangleIndices.cuptr(),
        nodes.cuptr(), n);

    if (n > 1) {
        gpu::WorkSize topologyWorkSize(GROUP_SIZE, n - 1);

        ::build_lbvh_topology<<<topologyWorkSize.cuGridSize(), topologyWorkSize.cuBlockSize(), 0, stream>>>(
            codes.cuptr(), nodes.cuptr(), parents.cuptr(), n);

        ready.resizeN(n - 1);

        CUDA_SAFE_CALL(cudaMemsetAsync(
            ready.cuptr(), 0,
            size_t(n - 1) * sizeof(unsigned int), stream));

        ::build_lbvh_bounds<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            nodes.cuptr(), parents.cuptr(), ready.cuptr(), n);
    }

    CUDA_CHECK_KERNEL_SYNC(stream);
}

}