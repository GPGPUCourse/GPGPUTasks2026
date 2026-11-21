#pragma once
#include <cuda_runtime.h>

namespace {
__device__ float3 loadVertex(const float* vertices, unsigned int vi)
{
    return {vertices[3 * vi + 0], vertices[3 * vi + 1], vertices[3 * vi + 2]};
}

__device__ uint3 loadFace(const unsigned int* faces, unsigned int fi)
{
    return {faces[3 * fi + 0], faces[3 * fi + 1], faces[3 * fi + 2]};
}
}
