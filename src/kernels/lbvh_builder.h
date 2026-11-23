#pragma once

#include "defines.h"
#include "kernels.h"
#include "shared_structs/bvh_node_gpu_shared.h"

namespace ocl {

class LBVHBuilder {
public:
    explicit LBVHBuilder(unsigned int count)
        : count(count),
          centroid_capacity(block_capacity(count, GROUP_SIZE)),
          tile_capacity(block_capacity(count, LBVH_TILE_SIZE)),
          keys_a(2 * size_t(count)), keys_b(2 * size_t(count)),
          bounds(2 * size_t(std::max(centroid_capacity, tile_capacity)))
    {}

    void build(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
               gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes,
               gpu::gpu_mem_32u& leaf_faces)
    {
        const gpu::WorkSize face_work(GROUP_SIZE, count);
        centroid_bounds.exec(gpu::WorkSize(GROUP_SIZE, size_t(centroid_capacity) * GROUP_SIZE),
                             vertices, faces, bounds.clmem(), centroid_capacity, count);
        reduce(centroid_capacity);

        morton_codes.exec(face_work, vertices, faces, bounds.clmem(), keys_a, count);
        auto* source = &keys_a;
        auto* destination = &keys_b;
        for (unsigned int run = 1; run < count; run *= 2) {
            merge_keys.exec(face_work, *source, *destination, run, count);
            std::swap(source, destination);
        }
        make_leaves.exec(face_work, vertices, faces, *source, nodes.clmem(), leaf_faces, count);
        if (count == 1)
            return;

        tile_bounds.exec(gpu::WorkSize(GROUP_SIZE, tile_capacity),
                         nodes.clmem(), bounds.clmem(), tile_capacity, count);
        reduce(tile_capacity);
        make_internal_nodes.exec(gpu::WorkSize(GROUP_SIZE, count - 1),
                                 *source, bounds.clmem(), nodes.clmem(), tile_capacity, count);
    }

private:
    static unsigned int block_capacity(unsigned int count, unsigned int block_size)
    {
        const unsigned int blocks = 1 + (count - 1) / block_size;
        unsigned int capacity = 1;
        while (capacity < blocks)
            capacity *= 2;
        return capacity;
    }

    void reduce(unsigned int capacity)
    {
        for (unsigned int level = capacity / 2; level > 0; level /= 2)
            reduce_boxes.exec(gpu::WorkSize(GROUP_SIZE, level), bounds.clmem(), level);
    }

    unsigned int count, centroid_capacity, tile_capacity;
    gpu::gpu_mem_32u keys_a, keys_b;
    gpu::shared_device_buffer_typed<AABBGPU> bounds;

    KernelSource centroid_bounds{getLBVHBuild(), "lbvh_centroid_bounds"};
    KernelSource reduce_boxes{getLBVHBuild(), "lbvh_reduce_boxes"};
    KernelSource morton_codes{getLBVHBuild(), "lbvh_morton_codes"};
    KernelSource merge_keys{getLBVHBuild(), "lbvh_merge_keys"};
    KernelSource make_leaves{getLBVHBuild(), "lbvh_make_leaves"};
    KernelSource tile_bounds{getLBVHBuild(), "lbvh_tile_bounds"};
    KernelSource make_internal_nodes{getLBVHBuild(), "lbvh_make_internal_nodes"};
};

} // namespace ocl
