#pragma once

#include <string>

#include "../kernels/shared_structs/camera_gpu_shared.h"

// Parse file containing <VCGCamera .../> and <ViewSettings .../>
CameraViewGPU loadViewState(const std::string& path);

// Parse from in-memory XML string (useful for tests)
CameraViewGPU parseViewStateFromString(const std::string& xml);

std::string dumpViewStateToString(const CameraViewGPU& camera);
