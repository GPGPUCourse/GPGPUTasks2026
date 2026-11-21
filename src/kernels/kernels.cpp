#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "vk/generated_kernels/aplusb_comp.h"

#include "cl/generated_kernels/sw_rasterize_normals.h"
#include "vk/generated_kernels/sw_rasterize_normals_comp.h"
#include "vk/generated_kernels/hw_rasterize_normals_vert.h"
#include "vk/generated_kernels/hw_rasterize_normals_frag.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
} // namespace cuda
#endif

namespace ocl {
const ProgramBinaries& getSWRasterizeNormals() { return opencl_binaries_sw_rasterize_normals; }

const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getSWRasterizeNormals() { return vulkan_binaries_sw_rasterize_normals_comp; }
std::vector<const ProgramBinaries*> getHWRasterizeNormals()
{
    return {&vulkan_binaries_hw_rasterize_normals_vert, &vulkan_binaries_hw_rasterize_normals_frag};
}

const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}
} // namespace avk2
