#include "kernels.h"
#include "defines.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/build_lbvh.h"
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

LBVHBuilder::LBVHBuilder(unsigned int nfaces)
    : nfaces_(nfaces),
      triangle_bounds_kernel_(opencl_binaries_build_lbvh, "lbvh_triangle_bounds"),
      reduce_bounds_kernel_(opencl_binaries_build_lbvh, "lbvh_reduce_bounds"),
      morton_keys_kernel_(opencl_binaries_build_lbvh, "lbvh_morton_keys"),
      sort_tiles_kernel_(opencl_binaries_build_lbvh, "lbvh_sort_tiles"),
      merge_tiles_kernel_(opencl_binaries_build_lbvh, "lbvh_merge_tiles"),
      topology_kernel_(opencl_binaries_build_lbvh, "lbvh_build_topology"),
      refit_kernel_(opencl_binaries_build_lbvh, "lbvh_refit_bounds")
{
    if (nfaces_ == 0)
        return;

    // Faces have three packed uints each. This also keeps topology search offsets in range.
    rassert(nfaces_ <= UINT32_MAX / 3, 384716201);
    const unsigned int groups = div_ceil(nfaces_, static_cast<unsigned int>(GROUP_SIZE));
    centroids_.resizeN(4ull * nfaces_);
    triangle_bounds_.resizeN(nfaces_);
    bounds_a_.resizeN(groups);
    keys_a_.resizeN(nfaces_);
    if (groups > 1)
        bounds_b_.resizeN(div_ceil(groups, static_cast<unsigned int>(GROUP_SIZE)));
    prefixes_.resizeN(std::max(1u, nfaces_ - 1));
    if (nfaces_ > 1)
        keys_b_.resizeN(nfaces_);

    triangle_bounds_kernel_.precompile();
    reduce_bounds_kernel_.precompile();
    morton_keys_kernel_.precompile();
    sort_tiles_kernel_.precompile();
    merge_tiles_kernel_.precompile();
    topology_kernel_.precompile();
    refit_kernel_.precompile();
}

void LBVHBuilder::build(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
                        gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes, gpu::gpu_mem_32u& leaf_triangles)
{
    if (nfaces_ == 0)
        return;

    const gpu::WorkSize triangle_work(GROUP_SIZE, nfaces_);
    triangle_bounds_kernel_.execAsynchronized(triangle_work, vertices, faces, centroids_,
                                             triangle_bounds_.clmem(), bounds_a_.clmem(), nfaces_);
    auto* bounds = &bounds_a_;
    auto* next_bounds = &bounds_b_;
    for (unsigned int count = div_ceil(nfaces_, static_cast<unsigned int>(GROUP_SIZE)); count > 1;
         count = div_ceil(count, static_cast<unsigned int>(GROUP_SIZE))) {
        reduce_bounds_kernel_.execAsynchronized(gpu::WorkSize(GROUP_SIZE, count), bounds->clmem(), next_bounds->clmem(), count);
        std::swap(bounds, next_bounds);
    }
    morton_keys_kernel_.execAsynchronized(triangle_work, centroids_, bounds->clmem(), keys_a_.clmem(), nfaces_);

    const int n = static_cast<int>(nfaces_);
    if (n == 1) {
        // The sole leaf is also the root.
        topology_kernel_.exec(triangle_work, keys_a_.clmem(), triangle_bounds_.clmem(),
                              nodes.clmem(), leaf_triangles, prefixes_, n);
        return;
    }

    sort_tiles_kernel_.execAsynchronized(triangle_work, keys_a_.clmem(), keys_b_.clmem(), nfaces_);
    auto* sorted = &keys_b_;
    auto* merge_output = &keys_a_;
    const gpu::WorkSize merge_work(GROUP_SIZE, div_ceil(nfaces_, LBVH_MERGE_ITEMS));
    for (unsigned int run = GROUP_SIZE; run < nfaces_; run *= 2) {
        merge_tiles_kernel_.execAsynchronized(merge_work, sorted->clmem(), merge_output->clmem(), run, nfaces_);
        std::swap(sorted, merge_output);
    }
    topology_kernel_.execAsynchronized(triangle_work, sorted->clmem(), triangle_bounds_.clmem(),
                                       nodes.clmem(), leaf_triangles, prefixes_, n);

    // Kernel completion on the in-order queue publishes the bounds from each band.
    const unsigned int leaf_start = nfaces_ - 1;
    const gpu::WorkSize refit_work(GROUP_SIZE, leaf_start);
    for (unsigned int prefix = 64 - LBVH_REFIT_PREFIX_BITS; prefix > 0; prefix -= LBVH_REFIT_PREFIX_BITS)
        refit_kernel_.execAsynchronized(refit_work, nodes.clmem(), prefixes_, leaf_start, prefix);
    refit_kernel_.exec(refit_work, nodes.clmem(), prefixes_, leaf_start, 0u);
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
