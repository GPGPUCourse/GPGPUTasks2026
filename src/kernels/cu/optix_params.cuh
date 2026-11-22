#pragma once

#include <optix.h>
#include "../shared_structs/camera_gpu_shared.h"

struct OptixRenderParams {
    OptixTraversableHandle scene;
    const float* vertices;
    const unsigned* faces;
    CameraViewGPU camera;
    int* faceIds;
    float* ao;
    float* depth;
};
