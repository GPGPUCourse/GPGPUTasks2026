#ifndef projection_gpu_shared_pragma_once
#define projection_gpu_shared_pragma_once

#include "struct_helpers.h"

// Row-major world-to-clip matrix. Vulkan uses the equivalent vec4 rows[4].
GPU_STRUCT_BEGIN(ProjectionGPU)
    float rows[4][4];
GPU_STRUCT_END(ProjectionGPU)

#endif
