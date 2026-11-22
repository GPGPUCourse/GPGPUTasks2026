#pragma once

#include <libgpu/vulkan/engine.h>

#include "shared_structs/camera_gpu_shared.h"
#include "shared_structs/bvh_node_gpu_shared.h"

namespace cuda {
bool render_optix(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const CameraViewGPU& camera, gpu::gpu_mem_32i& ids, gpu::gpu_mem_32f& ao, gpu::gpu_mem_32f& depth);
void build_lbvh(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes, gpu::gpu_mem_32u& leafFaces, int count);
void denoise_ao(const gpu::gpu_mem_32i& faces, const gpu::gpu_mem_32f& ao,
    gpu::gpu_mem_32f& filtered, int width, int height);
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
    unsigned int nfaces, gpu::gpu_mem_32f* depth = nullptr);
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
