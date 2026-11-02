#ifndef morton_code_gpu_shared_pragma_once // pragma once
#define morton_code_gpu_shared_pragma_once

#include "struct_helpers.h"

/* Language-agnostic 32-bit unsigned */
#if defined(__OPENCL_VERSION__)
  /* OpenCL C */
  #define MortonCode uint
#else
  /* C/C++/CUDA */
  #include <stdint.h>
  #define MortonCode uint32_t
#endif

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__)
  #if defined(__cplusplus)
    static_assert(sizeof(MortonCode) == 4, "MortonCode must be 32-bit");
  #endif
#endif

#endif // pragma once
