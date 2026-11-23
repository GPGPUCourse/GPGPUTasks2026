#pragma once

#include "kernels/shared_structs/aabb_gpu_shared.h"
#include "kernels/shared_structs/bvh_node_gpu_shared.h"
#include "kernels/defines.h"

#include <cstdint>
#include <algorithm>

namespace lbvh {

inline size_t round_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace lbvh


static inline bool build_lbvh_gpu(
    const gpu::gpu_mem_32f& vertices,
    const gpu::gpu_mem_32u& faces,
    uint32_t nfaces,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes_gpu,
    gpu::gpu_mem_32u& leaf_indices_gpu)
{
    if (nfaces == 0)
        return false;

    const size_t global_faces =
        lbvh::round_up(nfaces, GROUP_SIZE);

    const uint32_t num_groups =
        (nfaces + GROUP_SIZE - 1) / GROUP_SIZE;

    nodes_gpu.resizeN(2 * nfaces - 1);
    leaf_indices_gpu.resizeN(nfaces);

    gpu::shared_device_buffer_typed<AABBGPU> triangle_aabbs_gpu(nfaces);
    gpu::gpu_mem_32f centroids_gpu(3 * nfaces);

    gpu::gpu_mem_32f reduction_a_gpu(6 * num_groups);
    gpu::gpu_mem_32f reduction_b_gpu(6 * num_groups);

    gpu::gpu_mem_32u morton_a_gpu(nfaces);
    gpu::gpu_mem_32u morton_b_gpu(nfaces);

    gpu::gpu_mem_32u indices_a_gpu(nfaces);
    gpu::gpu_mem_32u indices_b_gpu(nfaces);

    gpu::gpu_mem_32u local_rank_gpu(nfaces);
    gpu::gpu_mem_32u histogram_gpu(16 * num_groups);
    gpu::gpu_mem_32u totals_gpu(16);

    const auto& program = ocl::getLBVHBuild();

    ocl::KernelSource k_triangle_aabb(
        program, "calculating_triangle_aabb");

    ocl::KernelSource k_local_reduction(
        program, "local_reduction");

    ocl::KernelSource k_global_reduction(
        program, "global_reduction");

    ocl::KernelSource k_morton(
        program, "morton_code");

    ocl::KernelSource k_init_indices(
        program, "initialize_triangle_indices");

    ocl::KernelSource k_radix_01(
        program, "radix_sort_01_local_counting");

    ocl::KernelSource k_radix_02(
        program, "radix_sort_02_global_prefixes_scan_sum_reduction");

    ocl::KernelSource k_radix_03(
        program, "radix_sort_03_global_prefixes_scan_accumulation");

    ocl::KernelSource k_radix_04(
        program, "radix_sort_04_scatter");

    ocl::KernelSource k_topology(
        program, "build_lbvh_topology");

    ocl::KernelSource k_init_leaves(
        program, "initialize_leaf_nodes");

    ocl::KernelSource k_compute_internal_aabbs(
        program, "compute_internal_aabbs");

    k_triangle_aabb.exec(
        gpu::WorkSize(GROUP_SIZE, global_faces),
        vertices,
        faces,
        nfaces,
        triangle_aabbs_gpu.clmem(),
        centroids_gpu);

    k_local_reduction.exec(
        gpu::WorkSize(GROUP_SIZE, global_faces),
        nfaces,
        centroids_gpu,
        reduction_a_gpu);

    gpu::gpu_mem_32f* reduction_src = &reduction_a_gpu;
    gpu::gpu_mem_32f* reduction_dst = &reduction_b_gpu;

    uint32_t reduction_groups = num_groups;

    while (reduction_groups > 1) {
        const uint32_t next_groups =
            (reduction_groups + GROUP_SIZE - 1) / GROUP_SIZE;

        const size_t global_reduction =
            lbvh::round_up(reduction_groups, GROUP_SIZE);

        k_global_reduction.exec(
            gpu::WorkSize(GROUP_SIZE, global_reduction),
            *reduction_src,
            reduction_groups,
            *reduction_dst);

        std::swap(reduction_src, reduction_dst);
        reduction_groups = next_groups;
    }

    k_morton.exec(
        gpu::WorkSize(GROUP_SIZE, global_faces),
        centroids_gpu,
        *reduction_src,
        nfaces,
        morton_a_gpu);

    k_init_indices.exec(
        gpu::WorkSize(GROUP_SIZE, global_faces),
        nfaces,
        indices_a_gpu);

    gpu::gpu_mem_32u* morton_src = &morton_a_gpu;
    gpu::gpu_mem_32u* morton_dst = &morton_b_gpu;

    gpu::gpu_mem_32u* indices_src = &indices_a_gpu;
    gpu::gpu_mem_32u* indices_dst = &indices_b_gpu;

    const gpu::WorkSize radix_work(
        GROUP_SIZE, global_faces);

    const gpu::WorkSize radix_small_work(
        GROUP_SIZE, GROUP_SIZE);

    for (uint32_t bit = 0; bit < 32; bit += 4) {
        k_radix_01.exec(
            radix_work,
            *morton_src,
            local_rank_gpu,
            histogram_gpu,
            nfaces,
            bit);

        k_radix_02.exec(
            radix_small_work,
            histogram_gpu,
            totals_gpu,
            nfaces);

        k_radix_03.exec(
            radix_small_work,
            histogram_gpu,
            totals_gpu,
            nfaces);

        k_radix_04.exec(
            radix_work,
            *morton_src,
            *indices_src,
            local_rank_gpu,
            histogram_gpu,
            *morton_dst,
            *indices_dst,
            nfaces,
            bit);

        std::swap(morton_src, morton_dst);
        std::swap(indices_src, indices_dst);
    }

    if (nfaces > 1) {
        const size_t global_internal =
            lbvh::round_up(nfaces - 1, GROUP_SIZE);

        k_topology.exec(
            gpu::WorkSize(GROUP_SIZE, global_internal),
            *morton_src,
            nfaces,
            nodes_gpu.clmem());

        k_init_leaves.exec(
            gpu::WorkSize(GROUP_SIZE, global_faces),
            nfaces,
            *indices_src,
            triangle_aabbs_gpu.clmem(),
            nodes_gpu.clmem());

        k_compute_internal_aabbs.exec(
            gpu::WorkSize(GROUP_SIZE, global_internal),
            *morton_src,
            nfaces,
            nodes_gpu.clmem());
    }
    else {
        k_init_leaves.exec(
            gpu::WorkSize(GROUP_SIZE, global_faces),
            nfaces,
            *indices_src,
            triangle_aabbs_gpu.clmem(),
            nodes_gpu.clmem());
    }

    // Copy sorted triangle indices into the output buffer.
    indices_src->copyToN(leaf_indices_gpu, nfaces);

    return true;
}
