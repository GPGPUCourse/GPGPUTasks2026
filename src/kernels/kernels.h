#pragma once

#include <libgpu/vulkan/engine.h>

#include "shared_structs/projection_gpu_shared.h"

struct SceneGeometry;

namespace cpu {
image8u sw_rasterize_normals(const SceneGeometry &scene, const ProjectionGPU &projection, int width, int height);
}

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void sw_rasterize_normals(const gpu::WorkSize &work_size, const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u &faces, const gpu::gpu_mem_64u &framebuffer,
    const ProjectionGPU &projection, unsigned int nfaces, int width, int height);
}

namespace ocl {
const ProgramBinaries& getAplusB();
const ProgramBinaries& getSWRasterizeNormals();
}

namespace avk2 {
const ProgramBinaries& getAplusB();
const ProgramBinaries& getSWRasterizeNormals();

std::vector<const ProgramBinaries*> getHWRasterizeNormals();
}
