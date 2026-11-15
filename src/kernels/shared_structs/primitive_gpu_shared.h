#ifndef primitive_gpu_shared_pragma_once // pragma once
#define primitive_gpu_shared_pragma_once

#include "struct_helpers.h"
#include "aabb_gpu_shared.h"

/* Language-agnostic 32-bit unsigned */
#if defined(__OPENCL_VERSION__)
  /* OpenCL C */
  #define GPUC_UINT uint
#elif defined(common_vk)
  /* Vulkan GLSL */
  #define GPUC_UINT uint
#else
  /* C/C++/CUDA */
  #include <stdint.h>
  #define GPUC_UINT uint32_t
#endif

// POD triangle primitive struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
GPU_STRUCT_BEGIN(PrimitiveGPU)
    GPUC_UINT triangle_index;
    GPUC_UINT morton_code;
    AABBGPU   aabb;
GPU_STRUCT_END(PrimitiveGPU)

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__) && !defined(common_vk)
  /* These static_asserts are ignored in OpenCL C.
     They guarantee identical, padding-free layout for host/CUDA. */
  #if defined(__cplusplus)
    static_assert(sizeof(GPUC_UINT) == 4, "GPUC_UINT must be 32-bit");

    static_assert(sizeof(float) == 4, "float must be 32-bit");

    static_assert(sizeof(PrimitiveGPU) == 4 + 4 + sizeof(AABBGPU), "Primitive size mismatch");
  #endif
#endif

#endif // pragma once
