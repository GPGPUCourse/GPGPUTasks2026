#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/lbvh_centroid_bounds.h"
#include "cl/generated_kernels/lbvh_scene_bounds.h"
#include "cl/generated_kernels/lbvh_morton_codes.h"
#include "cl/generated_kernels/lbvh_merge_pairs.h"
#include "cl/generated_kernels/lbvh_build_topology.h"
#include "cl/generated_kernels/lbvh_init_leaves.h"
#include "cl/generated_kernels/lbvh_build_aabbs.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/ray_tracing_render_brute_force_comp.h"
#include "vk/generated_kernels/ray_tracing_render_using_lbvh_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
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
const ProgramBinaries& getLBVHCentroidBounds()
{
    return opencl_binaries_lbvh_centroid_bounds;
}
const ProgramBinaries& getLBVHSceneBounds()
{
    return opencl_binaries_lbvh_scene_bounds;
}
const ProgramBinaries& getLBVHMortonCodes()
{
    return opencl_binaries_lbvh_morton_codes;
}
const ProgramBinaries& getLBVHMergePairs()
{
    return opencl_binaries_lbvh_merge_pairs;
}
const ProgramBinaries& getLBVHBuildTopology()
{
    return opencl_binaries_lbvh_build_topology;
}
const ProgramBinaries& getLBVHInitLeaves()
{
    return opencl_binaries_lbvh_init_leaves;
}
const ProgramBinaries& getLBVHBuildAABBs()
{
    return opencl_binaries_lbvh_build_aabbs;
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
