#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"
#include "cl/generated_kernels/compute_primitive_data.h"
#include "cl/generated_kernels/reduce_centroid_bounds.h"
#include "cl/generated_kernels/generate_morton_codes.h"
#include "cl/generated_kernels/fill_buffer_with_zeros.h"
#include "cl/generated_kernels/radix_sort_01_local_counting.h"
#include "cl/generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction.h"
#include "cl/generated_kernels/radix_sort_03_scatter_kv.h"
#include "cl/generated_kernels/build_lbvh_topology_and_leaves.h"
#include "cl/generated_kernels/refit_internal_aabbs.h"

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

const ProgramBinaries& getComputePrimitiveData()
{
    return opencl_binaries_compute_primitive_data;
}

const ProgramBinaries& getReduceCentroidBounds()
{
    return opencl_binaries_reduce_centroid_bounds;
}

const ProgramBinaries& getGenerateMortonCodes()
{
    return opencl_binaries_generate_morton_codes;
}

const ProgramBinaries& getFillBufferWithZeros()
{
    return opencl_binaries_fill_buffer_with_zeros;
}

const ProgramBinaries& getRadixSort01LocalCounting()
{
    return opencl_binaries_radix_sort_01_local_counting;
}

const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction()
{
    return opencl_binaries_radix_sort_02_global_prefixes_scan_sum_reduction;
}

const ProgramBinaries& getRadixSort03Scatter()
{
    return opencl_binaries_radix_sort_03_scatter_kv;
}

const ProgramBinaries& getBuildLBVHTopologyAndLeaves()
{
    return opencl_binaries_build_lbvh_topology_and_leaves;
}

const ProgramBinaries& getRefitInternalAabbs()
{
    return opencl_binaries_refit_internal_aabbs;
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
