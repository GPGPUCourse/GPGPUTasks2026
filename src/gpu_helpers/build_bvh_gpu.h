#pragma once

#include "../cpu_helpers/morton_code_cpu.h"
#include "../kernels/shared_structs/aabb_gpu_shared.h"
#include "../kernels/shared_structs/bvh_node_gpu_shared.h"

#include <libbase/point.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace albuquerque {

// count leading zeros for 32-bit unsigned
static inline int clz32(uint32_t x)
{
    if (x == 0u)
        return 32;
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, x);
    return 31 - static_cast<int>(idx);
#else
    return __builtin_clz(x);
#endif
}

// Compute common prefix length between sorted Morton codes at indices i and j
// Uses index as a tiebreaker when codes are equal (like Karras 2013).
static inline int common_prefix(const std::vector<MortonCode>& codes, int N, int i, int j)
{
    if (j < 0 || j >= N)
        return -1;

    MortonCode ci = codes[static_cast<size_t>(i)];
    MortonCode cj = codes[static_cast<size_t>(j)];

    if (ci == cj) {
        uint32_t di = static_cast<uint32_t>(i);
        uint32_t dj = static_cast<uint32_t>(j);
        uint32_t diff = di ^ dj;
        return 32 + clz32(diff);
    } else {
        uint32_t diff = ci ^ cj;
        return clz32(diff);
    }
}

// Determine range [first, last] of primitives covered by internal node i
static inline void determine_range(const std::vector<uint32_t>& codes, int N, int i, int& outFirst, int& outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    outFirst = std::min(i, j);
    outLast = std::max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
static inline int find_split(const std::vector<uint32_t>& codes,
    int first, int last)
{
    const int N = static_cast<int>(codes.size());

    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    auto delta = [&](int i, int j) -> int {
        return common_prefix(codes, N, i, j);
    };

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = delta(first, last);

    int split = first;
    int step = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = delta(first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

// Sorts (keys_a, vals_a) by key, ascending. Uses (keys_b, vals_b) as scratch.
// Morton codes are 30 bits -> 30 passes (even), so the result ends up in the *_a buffers.
inline void radixSortKV_GPU(
    gpu::gpu_mem_32u& keys_a, gpu::gpu_mem_32u& vals_a,
    gpu::gpu_mem_32u& keys_b, gpu::gpu_mem_32u& vals_b,
    unsigned int n)
{
    static ocl::KernelSource ocl_scan(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    static ocl::KernelSource ocl_scatter_kv(ocl::getRadixSort03Scatter());

    const unsigned int ITEMS_PER_THREAD = 8; // must match the .cl
    const unsigned int tile_size = GROUP_SIZE * ITEMS_PER_THREAD;
    const unsigned int num_tiles = (n + tile_size - 1) / tile_size;

    gpu::gpu_mem_32u scan(num_tiles * tile_size); // padded: the kernel does full vstore8
    gpu::gpu_mem_32u spine(2u * num_tiles);
    gpu::gpu_mem_32u tile_bump(1);
    gpu::gpu_mem_32u total(1);

    const gpu::WorkSize scan_ws(GROUP_SIZE, num_tiles * GROUP_SIZE);
    const gpu::WorkSize scatter_ws(GROUP_SIZE, (n + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE);

    gpu::gpu_mem_32u* keys_src = &keys_a;
    gpu::gpu_mem_32u* vals_src = &vals_a;
    gpu::gpu_mem_32u* keys_dst = &keys_b;
    gpu::gpu_mem_32u* vals_dst = &vals_b;

    for (unsigned int bit = 0; bit < 30; ++bit) {
        spine.fill(0u);
        tile_bump.fill(0u);

        ocl_scan.exec(scan_ws, *keys_src, scan, spine, tile_bump, total, n, bit);
        ocl_scatter_kv.exec(scatter_ws, *keys_src, *vals_src, scan, total, *keys_dst, *vals_dst, n, bit);

        std::swap(keys_src, keys_dst);
        std::swap(vals_src, vals_dst);
    }
}

// Build LBVH (Karras 2013) on CPU.
// Output:
//   outNodes           - BVH nodes array of size (2*N - 1). Root is node 0.
//   outLeafTriIndices  - size N, mapping leaf i -> original triangle index.
//
// Node indexing convention (matches LBVH style):
//   N = number of triangles (faces.size()).
//   Internal nodes: indices [0 .. N-2]
//   Leaf nodes:     indices [N-1 .. 2*N-2]
//   Leaf at index (N-1 + i) corresponds to outLeafTriIndices[i].
inline void buildLBVH_GPU(
    const std::vector<point3f>& vertices,
    const std::vector<point3u>& faces,
    std::vector<BVHNodeGPU>& outNodes,
    std::vector<uint32_t>& outLeafTriIndices)
{
    static ocl::KernelSource ocl_compute_primitive_data(ocl::getComputePrimitiveData());
    static ocl::KernelSource ocl_reduce_centroid_bounds(ocl::getReduceCentroidBounds());
    static ocl::KernelSource ocl_generate_morton_codes(ocl::getGenerateMortonCodes());

    const size_t N = faces.size();
    outNodes.clear();
    outLeafTriIndices.clear();

    if (N == 0) {
        return;
    }

    // Special case: single triangle -> single leaf/root
    if (N == 1) {
        outNodes.resize(1);
        outLeafTriIndices.resize(1);

        const point3u& f = faces[0];
        const point3f& v0 = vertices[f.x];
        const point3f& v1 = vertices[f.y];
        const point3f& v2 = vertices[f.z];

        AABBGPU aabb;
        aabb.min_x = std::min({ v0.x, v1.x, v2.x });
        aabb.min_y = std::min({ v0.y, v1.y, v2.y });
        aabb.min_z = std::min({ v0.z, v1.z, v2.z });
        aabb.max_x = std::max({ v0.x, v1.x, v2.x });
        aabb.max_y = std::max({ v0.y, v1.y, v2.y });
        aabb.max_z = std::max({ v0.z, v1.z, v2.z });

        BVHNodeGPU& node = outNodes[0];
        node.aabb = aabb;
        // Leaf: no children (user can detect leaf via index and N)
        node.leftChildIndex = std::numeric_limits<GPUC_UINT>::max();
        node.rightChildIndex = std::numeric_limits<GPUC_UINT>::max();

        outLeafTriIndices[0] = 0;
        return;
    }

    // Per-triangle info
    struct Prim {
        uint32_t triIndex;
        uint32_t morton;
        AABBGPU aabb;
        point3f centroid;
    };

    const unsigned int WG = GROUP_SIZE;
    const unsigned int numGroups = (N + WG - 1) / WG;
    const unsigned int globalSize = numGroups * WG;

    // 1) GPU: per-triangle AABB + centroid, and global centroid bounds
    gpu::gpu_mem_32f vertices_gpu(3 * vertices.size());
    gpu::gpu_mem_32u faces_gpu(3 * faces.size());
    gpu::shared_device_buffer_typed<AABBGPU> prim_aabbs_gpu(N);
    gpu::gpu_mem_32f centroids_gpu(3 * N);
    gpu::gpu_mem_32f partial_min_gpu(3 * numGroups);
    gpu::gpu_mem_32f partial_max_gpu(3 * numGroups);
    gpu::gpu_mem_32f centroid_bounds_gpu(6); // [0..2] = min, [3..5] = max
    vertices_gpu.writeN((const float*)vertices.data(), 3 * vertices.size());
    faces_gpu.writeN((const unsigned int*)faces.data(), 3 * faces.size());

    // vertices_gpu / faces_gpu are the buffers you already uploaded
    // (declare these kernels the same way as ocl_rt_brute_force)
    ocl_compute_primitive_data.exec(
        gpu::WorkSize(WG, globalSize),
        vertices_gpu, faces_gpu,
        prim_aabbs_gpu.clmem(), centroids_gpu,
        partial_min_gpu, partial_max_gpu,
        (unsigned int)N);

    ocl_reduce_centroid_bounds.exec(
        gpu::WorkSize(WG, WG), // a single workgroup
        partial_min_gpu, partial_max_gpu,
        centroid_bounds_gpu,
        numGroups);

    // Read back (temporary, until the later stages move to the GPU)
    std::vector<AABBGPU> gpu_aabbs(N);
    std::vector<float> gpu_centroids(3 * N);
    float gpu_bounds[6];
    prim_aabbs_gpu.readN(gpu_aabbs.data(), N);
    centroids_gpu.readN(gpu_centroids.data(), 3 * N);
    centroid_bounds_gpu.readN(gpu_bounds, 6);

    // Fill the same structures the CPU code already uses
    std::vector<Prim> prims(N);
    for (size_t i = 0; i < N; ++i) {
        prims[i].triIndex = static_cast<uint32_t>(i);
        prims[i].aabb = gpu_aabbs[i];
        prims[i].centroid = { gpu_centroids[3 * i + 0],
            gpu_centroids[3 * i + 1],
            gpu_centroids[3 * i + 2] };
    }
    point3f cMin { gpu_bounds[0], gpu_bounds[1], gpu_bounds[2] };
    point3f cMax { gpu_bounds[3], gpu_bounds[4], gpu_bounds[5] };

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    gpu::gpu_mem_32u morton_codes_gpu(N);
    gpu::gpu_mem_32u tri_indices_gpu(N);

    ocl_generate_morton_codes.exec(
        gpu::WorkSize(WG, globalSize),
        centroids_gpu, centroid_bounds_gpu,
        morton_codes_gpu, tri_indices_gpu,
        (unsigned int)N);

    std::vector<uint32_t> gpu_morton(N);
    morton_codes_gpu.readN(gpu_morton.data(), N);
    for (size_t i = 0; i < N; ++i) {
        prims[i].morton = gpu_morton[i];
    }

    // 3) Sort primitives by Morton code
    gpu::gpu_mem_32u morton_tmp_gpu(N);
    gpu::gpu_mem_32u tri_tmp_gpu(N);
    radixSortKV_GPU(morton_codes_gpu, tri_indices_gpu, morton_tmp_gpu, tri_tmp_gpu, N);

    std::vector<uint32_t> sorted_morton(N), sorted_tri(N);
    morton_codes_gpu.readN(sorted_morton.data(), N);
    tri_indices_gpu.readN(sorted_tri.data(), N);

    // replaces the CPU Morton loop and the std::sort
    std::vector<Prim> sorted(N);
    for (size_t i = 0; i < N; ++i) {
        sorted[i].triIndex = sorted_tri[i];
        sorted[i].morton = sorted_morton[i];
        sorted[i].aabb = gpu_aabbs[sorted_tri[i]];
    }
    prims.swap(sorted);

    // 4) Prepare arrays
    const size_t numNodes = 2 * N - 1;
    outNodes.resize(numNodes);
    outLeafTriIndices.resize(N);

    std::vector<uint32_t> sortedCodes(N);
    for (size_t i = 0; i < N; ++i) {
        sortedCodes[i] = prims[i].morton;
        outLeafTriIndices[i] = prims[i].triIndex;
    }

    // 5) Initialize leaf nodes [N-1 .. 2*N-2]
    const GPUC_UINT INVALID = std::numeric_limits<GPUC_UINT>::max();

    for (size_t i = 0; i < N; ++i) {
        size_t leafIndex = (N - 1) + i;
        BVHNodeGPU& leaf = outNodes[leafIndex];

        leaf.aabb = prims[i].aabb;
        leaf.leftChildIndex = INVALID;
        leaf.rightChildIndex = INVALID;
    }

    // 6) Build internal nodes [0 .. N-2]
    for (int i = 0; i < static_cast<int>(N) - 1; ++i) {
        int first, last;
        determine_range(sortedCodes, static_cast<int>(N), i, first, last);
        int split = find_split(sortedCodes, first, last);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = static_cast<int>((N - 1) + split);
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = static_cast<int>((N - 1) + split + 1);
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        node.leftChildIndex = static_cast<GPUC_UINT>(leftIndex);
        node.rightChildIndex = static_cast<GPUC_UINT>(rightIndex);
    }

    // 7.1) Enumerating bottom-up traversal order for internal nodes to build AABB
    std::vector<int> traversalOrder = { 0 }; // starting with root node
    std::set<int> alreadyVisited;
    size_t next = 0;
    while (next < traversalOrder.size()) {
        int i = traversalOrder[next++];
        rassert(alreadyVisited.count(i) == 0, 452341233412, i, next);
        alreadyVisited.insert(i);
        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        if (node.leftChildIndex != INVALID && node.leftChildIndex < N - 1) { // ensuring it is not a leaf
            traversalOrder.push_back(node.leftChildIndex);
        }
        if (node.rightChildIndex != INVALID && node.rightChildIndex < N - 1) {
            traversalOrder.push_back(node.rightChildIndex);
        }
    }
    rassert(traversalOrder.size() == N - 1, 354623412341, traversalOrder.size(), N);

    // 7.2) AABB propagation for internal nodes in the bottom-up traversal order
    for (int j = traversalOrder.size() - 1; j >= 0; --j) {
        int i = traversalOrder[j];
        rassert(i < outNodes.size(), 45123413211);
        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        const BVHNodeGPU& left = outNodes[static_cast<size_t>(node.leftChildIndex)];
        const BVHNodeGPU& right = outNodes[static_cast<size_t>(node.rightChildIndex)];

        AABBGPU aabb;
        aabb.min_x = std::min(left.aabb.min_x, right.aabb.min_x);
        aabb.min_y = std::min(left.aabb.min_y, right.aabb.min_y);
        aabb.min_z = std::min(left.aabb.min_z, right.aabb.min_z);
        aabb.max_x = std::max(left.aabb.max_x, right.aabb.max_x);
        aabb.max_y = std::max(left.aabb.max_y, right.aabb.max_y);
        aabb.max_z = std::max(left.aabb.max_z, right.aabb.max_z);

        node.aabb = aabb;
    }
}

}