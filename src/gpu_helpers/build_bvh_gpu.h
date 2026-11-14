#pragma once

#include "../kernels/shared_structs/aabb_gpu_shared.h"
#include "../kernels/shared_structs/bvh_node_gpu_shared.h"

#include <libbase/point.h>

#include <algorithm>
#include <vector>

namespace albuquerque {

static inline gpu::WorkSize workSize1D(unsigned int n)
{
    return gpu::WorkSize(GROUP_SIZE, (n + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE);
}

inline void radixSortKV_GPU(gpu::gpu_mem_32u& keys, gpu::gpu_mem_32u& vals, unsigned int n)
{
    static ocl::KernelSource ocl_count(ocl::getRadixSort01LocalCounting());
    static ocl::KernelSource ocl_scan(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    static ocl::KernelSource ocl_scatter(ocl::getRadixSort03Scatter());

    constexpr unsigned int PASSES = (30 + RADIX_BITS - 1) / RADIX_BITS;

    constexpr unsigned int tile_size = GROUP_SIZE * RADIX_ITEMS;
    const unsigned int num_tiles = (n + tile_size - 1) / tile_size;

    gpu::gpu_mem_32u keys_tmp(n), vals_tmp(n), hist(RADIX * num_tiles);
    const gpu::WorkSize tiles_ws(GROUP_SIZE, num_tiles * GROUP_SIZE);
    const gpu::WorkSize one_group(GROUP_SIZE, GROUP_SIZE);

    gpu::gpu_mem_32u *ks = &keys, *vs = &vals, *kd = &keys_tmp, *vd = &vals_tmp;
    for (unsigned int pass = 0; pass < PASSES; ++pass) {
        const unsigned int shift = pass * RADIX_BITS;
        ocl_count.exec(tiles_ws, *ks, hist, n, shift, num_tiles);
        ocl_scan.exec(one_group, hist, RADIX * num_tiles);
        ocl_scatter.exec(tiles_ws, *ks, *vs, hist, *kd, *vd, n, shift, num_tiles);
        std::swap(ks, kd);
        std::swap(vs, vd);
    }
}

inline void buildLBVH_GPU(
    const gpu::gpu_mem_32f& vertices_gpu,
    const gpu::gpu_mem_32u& faces_gpu,
    unsigned int n,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes_gpu,
    gpu::gpu_mem_32u& leaf_tri_indices_gpu)
{
    static ocl::KernelSource ocl_primitive_data(ocl::getComputePrimitiveData());
    static ocl::KernelSource ocl_centroid_bounds(ocl::getReduceCentroidBounds());
    static ocl::KernelSource ocl_morton(ocl::getGenerateMortonCodes());
    static ocl::KernelSource ocl_topology(ocl::getBuildLBVHTopologyAndLeaves());
    static ocl::KernelSource ocl_refit(ocl::getRefitInternalAabbs());

    if (n == 0)
        return;

    const unsigned int num_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    const unsigned int num_nodes = 2 * n - 1;

    gpu::shared_device_buffer_typed<AABBGPU> prim_aabbs_gpu(n);
    gpu::gpu_mem_32f centroids_gpu(3 * n);
    gpu::gpu_mem_32f partial_min_gpu(3 * num_groups), partial_max_gpu(3 * num_groups);
    gpu::gpu_mem_32f centroid_bounds_gpu(6);
    gpu::gpu_mem_32u morton_gpu(n);
    gpu::gpu_mem_32u parent_gpu(2 * num_nodes), arrival_gpu(std::max(n - 1, 1u));
    arrival_gpu.fill(0u);

    ocl_primitive_data.exec(workSize1D(n), vertices_gpu, faces_gpu,
        prim_aabbs_gpu.clmem(), centroids_gpu, partial_min_gpu, partial_max_gpu, n);
    ocl_centroid_bounds.exec(gpu::WorkSize(GROUP_SIZE, GROUP_SIZE),
        partial_min_gpu, partial_max_gpu, centroid_bounds_gpu, num_groups);

    // the Morton kernel initializes leaf_tri_indices_gpu to the identity; the sort permutes it
    ocl_morton.exec(workSize1D(n), centroids_gpu, centroid_bounds_gpu, morton_gpu, leaf_tri_indices_gpu, n);
    radixSortKV_GPU(morton_gpu, leaf_tri_indices_gpu, n);

    ocl_topology.exec(workSize1D(num_nodes), morton_gpu, leaf_tri_indices_gpu,
        prim_aabbs_gpu.clmem(), nodes_gpu.clmem(), parent_gpu, n);

    if (n > 1)
        ocl_refit.exec(workSize1D(n), nodes_gpu.clmem(), parent_gpu, arrival_gpu, n);
}

}