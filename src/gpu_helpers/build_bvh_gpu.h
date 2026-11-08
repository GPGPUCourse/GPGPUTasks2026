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

inline void buildLBVH_GPU(
    const std::vector<point3f>& vertices,
    const std::vector<point3u>& faces,
    std::vector<BVHNodeGPU>& outNodes,
    std::vector<uint32_t>& outLeafTriIndices)
{
    static ocl::KernelSource ocl_compute_primitive_data(ocl::getComputePrimitiveData());
    static ocl::KernelSource ocl_reduce_centroid_bounds(ocl::getReduceCentroidBounds());
    static ocl::KernelSource ocl_generate_morton_codes(ocl::getGenerateMortonCodes());
    static ocl::KernelSource ocl_build_topology(ocl::getBuildLBVHTopologyAndLeaves());
    static ocl::KernelSource ocl_refit(ocl::getRefitInternalAabbs());

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

    ocl_compute_primitive_data.exec(
        gpu::WorkSize(WG, globalSize),
        vertices_gpu, faces_gpu,
        prim_aabbs_gpu.clmem(), centroids_gpu,
        partial_min_gpu, partial_max_gpu,
        (unsigned int)N);

    ocl_reduce_centroid_bounds.exec(
        gpu::WorkSize(WG, WG),
        partial_min_gpu, partial_max_gpu,
        centroid_bounds_gpu,
        numGroups);

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    gpu::gpu_mem_32u morton_codes_gpu(N);
    gpu::gpu_mem_32u tri_indices_gpu(N);

    ocl_generate_morton_codes.exec(
        gpu::WorkSize(WG, globalSize),
        centroids_gpu, centroid_bounds_gpu,
        morton_codes_gpu, tri_indices_gpu,
        (unsigned int)N);

    // 3) Sort primitives by Morton code
    gpu::gpu_mem_32u morton_tmp_gpu(N);
    gpu::gpu_mem_32u tri_tmp_gpu(N);
    radixSortKV_GPU(morton_codes_gpu, tri_indices_gpu, morton_tmp_gpu, tri_tmp_gpu, N);

    // 4-6) Leaves + topology
    // 7.1) Enumerating bottom-up traversal order for internal nodes to build AABB
    // 7.2) AABB propagation for internal nodes in the bottom-up traversal order
    const size_t numNodes = 2 * N - 1;
    gpu::shared_device_buffer_typed<BVHNodeGPU> nodes_gpu(numNodes);
    gpu::gpu_mem_32u parent_gpu(numNodes);
    gpu::gpu_mem_32u arrival_gpu(N - 1);
    arrival_gpu.fill(0u);

    ocl_build_topology.exec(
        gpu::WorkSize(WG, (numNodes + WG - 1) / WG * WG),
        morton_codes_gpu, tri_indices_gpu,
        prim_aabbs_gpu.clmem(), nodes_gpu.clmem(),
        parent_gpu, // NEW
        (unsigned int)N);

    ocl_refit.exec(
        gpu::WorkSize(WG, globalSize), // one work-item per leaf (N, rounded up)
        nodes_gpu.clmem(), parent_gpu, arrival_gpu,
        (unsigned int)N);

    outNodes.resize(numNodes);
    outLeafTriIndices.resize(N);
    nodes_gpu.readN(outNodes.data(), numNodes);
    tri_indices_gpu.readN(outLeafTriIndices.data(), N);
}
}