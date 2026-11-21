#pragma once

#include <libgpu/vulkan/engine.h>

#include "shared_structs/bvh_node_gpu_shared.h"
#include "shared_structs/camera_gpu_shared.h"

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void ray_tracing_render_brute_force(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::gpu_mem_32i& framebuffer_face_id,
    gpu::gpu_mem_32f& framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);
void ray_tracing_render_using_lbvh(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU>& bvhNodes, const gpu::gpu_mem_32u& leafTriIndices,
    gpu::gpu_mem_32i& framebuffer_face_id,
    gpu::gpu_mem_32f& framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);

void faces_morton_codes(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const gpu::shared_device_buffer_typed<uint2>& morton_codes,
    unsigned int nfaces);

void sortMortonCodes(const gpu::shared_device_buffer_typed<uint2>& morton_codes, std::size_t count);

void build_lbvh(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices,
    const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<uint2>& mortonCodes,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& outNodes,
    gpu::gpu_mem_32u& outLeafTriIndices,
    gpu::gpu_mem_32u& parents,
    gpu::gpu_mem_32u& flags,
    unsigned int nfaces);

void propagate_aabb_in_lbvh(
    const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& outNodes,
    const gpu::gpu_mem_32u& parents,
    gpu::gpu_mem_32u& flags,
    unsigned int nfaces);
}



namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getRTBruteForce();
const ProgramBinaries& getRTWithLBVH();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getRTBruteForce();
const ProgramBinaries& getRTWithLBVH();
}
