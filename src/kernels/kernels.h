#pragma once

#include <libgpu/vulkan/engine.h>

#include <memory>

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

class TiledRasterizer {
public:
    TiledRasterizer(unsigned nfaces, int width, int height);
    ~TiledRasterizer();
    // The framebuffer must already be cleared to ~uint64_t{0}. Untouched tiles stay clear.
    void drawCleared(const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
                     const gpu::gpu_mem_64u &framebuffer, const ProjectionGPU &projection);

    struct Stats {
        unsigned triangles, activeTiles;
        unsigned long long binRefs, tileRefs;
        bool fallback;
    };
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
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
