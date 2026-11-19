#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/ray_tracing_render_brute_force_comp.h"
#include "vk/generated_kernels/ray_tracing_render_using_lbvh_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void lbvh_morton_codes(
    const gpu::gpu_mem_32f &centroids, const gpu::shared_device_buffer_typed<AABBGPU> &centroidBounds,
    unsigned int nfaces, gpu::gpu_mem_32u &mortonCodes, gpu::gpu_mem_32u &triangleIndices)
{
    rassert(false, 835175864);
}

void lbvh_initialize_leaves(
    const gpu::gpu_mem_32u &sortedIndices, const gpu::shared_device_buffer_typed<AABBGPU> &triangleBoxes,
    unsigned int nfaces, gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &parents)
{
    rassert(false, 835175865);
}

void lbvh_build_boxes(
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, const gpu::gpu_mem_32u &parents,
    gpu::gpu_mem_32u &arrivals, unsigned int nfaces)
{
    rassert(false, 835175863);
}

void lbvh_build_internal_nodes(
    const gpu::gpu_mem_32u &sortedCodes, unsigned int nfaces,
    gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes, gpu::gpu_mem_32u &parents)
{
    rassert(false, 835175857);
}

void lbvh_sort_morton_codes(
    const gpu::gpu_mem_32u &mortonCodes, const gpu::gpu_mem_32u &triangleIndices,
    unsigned int nfaces, gpu::gpu_mem_32u &sortedCodes, gpu::gpu_mem_32u &sortedIndices,
    gpu::shared_device_buffer &workspace)
{
    rassert(false, 835175851);
}

void lbvh_triangle_geometry(
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces, unsigned int nfaces,
    gpu::gpu_mem_32f &centroids, gpu::shared_device_buffer_typed<AABBGPU> &triangleBoxes,
    gpu::shared_device_buffer_typed<AABBGPU> &centroidBounds)
{
    rassert(false, 835175847);
}

void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void ray_tracing_render_brute_force(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getRTBruteForce()
{
    return opencl_binaries_ray_tracing_render_brute_force;
}

const ProgramBinaries& getRTWithLBVH()
{
    return opencl_binaries_ray_tracing_render_using_lbvh;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getRTBruteForce()
{
    return vulkan_binaries_ray_tracing_render_brute_force_comp;
}

const ProgramBinaries& getRTWithLBVH()
{
    return vulkan_binaries_ray_tracing_render_using_lbvh_comp;
}
} // namespace avk2
