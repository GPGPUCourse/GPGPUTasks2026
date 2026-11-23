#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/ray_tracing_render_brute_force_comp.h"
#include "vk/generated_kernels/ray_tracing_render_using_lbvh_comp.h"
#include "vk/generated_kernels/lbvh_prepare_comp.h"
#include "vk/generated_kernels/lbvh_triangle_info_comp.h"
#include "vk/generated_kernels/lbvh_centroid_bounds_partial_comp.h"
#include "vk/generated_kernels/lbvh_centroid_bounds_final_comp.h"
#include "vk/generated_kernels/lbvh_morton_codes_comp.h"
#include "vk/generated_kernels/radix_sort_01_local_counting_comp.h"
#include "vk/generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_comp.h"
#include "vk/generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_comp.h"
#include "vk/generated_kernels/radix_sort_04_scatter_comp.h"
#include "vk/generated_kernels/lbvh_init_leaves_comp.h"
#include "vk/generated_kernels/lbvh_topology_comp.h"
#include "vk/generated_kernels/lbvh_aabbs_comp.h"

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

const ProgramBinaries& getLBVHPrepare()
{
    return vulkan_binaries_lbvh_prepare_comp;
}

const ProgramBinaries& getLBVHTriangleInfo()
{
    return vulkan_binaries_lbvh_triangle_info_comp;
}

const ProgramBinaries& getLBVHCentroidBoundsPartial()
{
    return vulkan_binaries_lbvh_centroid_bounds_partial_comp;
}

const ProgramBinaries& getLBVHCentroidBoundsFinal()
{
    return vulkan_binaries_lbvh_centroid_bounds_final_comp;
}

const ProgramBinaries& getLBVHMortonCodes()
{
    return vulkan_binaries_lbvh_morton_codes_comp;
}

const ProgramBinaries& getRadixSort01LocalCounting()
{
    return vulkan_binaries_radix_sort_01_local_counting_comp;
}

const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction()
{
    return vulkan_binaries_radix_sort_02_global_prefixes_scan_sum_reduction_comp;
}

const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation()
{
    return vulkan_binaries_radix_sort_03_global_prefixes_scan_accumulation_comp;
}

const ProgramBinaries& getRadixSort04Scatter()
{
    return vulkan_binaries_radix_sort_04_scatter_comp;
}

const ProgramBinaries& getLBVHInitLeaves()
{
    return vulkan_binaries_lbvh_init_leaves_comp;
}

const ProgramBinaries& getLBVHTopology()
{
    return vulkan_binaries_lbvh_topology_comp;
}

const ProgramBinaries& getLBVHAABBs()
{
    return vulkan_binaries_lbvh_aabbs_comp;
}
} // namespace avk2
