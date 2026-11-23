#pragma once

#include <libgpu/opencl/engine.h>
#include <libgpu/vulkan/engine.h>

#include "shared_structs/camera_gpu_shared.h"
#include "shared_structs/bvh_node_gpu_shared.h"

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void ray_tracing_render_brute_force(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getRTBruteForce();
const ProgramBinaries& getRTWithLBVH();

class LBVHBuilder {
public:
    explicit LBVHBuilder(unsigned int nfaces);

    void build(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
               gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes, gpu::gpu_mem_32u& leaf_triangles);

private:
    unsigned int nfaces_;
    gpu::gpu_mem_32f centroids_;
    gpu::shared_device_buffer_typed<AABBGPU> triangle_bounds_;
    gpu::shared_device_buffer_typed<AABBGPU> bounds_a_, bounds_b_;
    gpu::shared_device_buffer_typed<uint64_t> keys_a_, keys_b_;
    gpu::gpu_mem_8u prefixes_;

    KernelSource triangle_bounds_kernel_, reduce_bounds_kernel_, morton_keys_kernel_;
    KernelSource sort_tiles_kernel_, merge_tiles_kernel_, topology_kernel_, refit_kernel_;
};
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getRTBruteForce();
const ProgramBinaries& getRTWithLBVH();
}
