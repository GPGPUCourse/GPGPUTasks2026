#pragma once

#include <libgpu/vulkan/engine.h>

#include "shared_structs/camera_gpu_shared.h"
#include "shared_structs/bvh_node_gpu_shared.h"

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void lbvh_morton_codes(
    const gpu::gpu_mem_32f &centroids, const gpu::shared_device_buffer_typed<AABBGPU> &centroidBounds,
    unsigned int nfaces, gpu::gpu_mem_32u &mortonCodes, gpu::gpu_mem_32u &triangleIndices);

void lbvh_initialize_leaves(
    const gpu::gpu_mem_32u &sortedIndices, const gpu::shared_device_buffer_typed<AABBGPU> &triangleBoxes,
    unsigned int nfaces, gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &parents);

void lbvh_build_boxes(
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, const gpu::gpu_mem_32u &parents,
    gpu::gpu_mem_32u &arrivals, unsigned int nfaces);

void lbvh_build_internal_nodes(
    const gpu::gpu_mem_32u &sortedCodes, unsigned int nfaces,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &parents);

void lbvh_sort_morton_codes(
    const gpu::gpu_mem_32u &mortonCodes, const gpu::gpu_mem_32u &triangleIndices,
    unsigned int nfaces, gpu::gpu_mem_32u &sortedCodes, gpu::gpu_mem_32u &sortedIndices,
    gpu::shared_device_buffer &workspace);

void lbvh_triangle_geometry(
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces, unsigned int nfaces,
    gpu::gpu_mem_32f &centroids, gpu::shared_device_buffer_typed<AABBGPU> &triangleBoxes,
    gpu::shared_device_buffer_typed<AABBGPU> &centroidBounds);

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
    unsigned int nfaces, gpu::gpu_mem_32f *depth = nullptr,
    gpu::gpu_mem_32f *normals = nullptr);

void denoise_ambient_occlusion(
    const gpu::gpu_mem_32f &ao, const gpu::gpu_mem_32f &depth,
    const gpu::gpu_mem_32f &normals, gpu::gpu_mem_32f &output,
    unsigned int width, unsigned int height, unsigned int radius,
    float spatialSigma, float depthSigma, float normalSigma);

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
