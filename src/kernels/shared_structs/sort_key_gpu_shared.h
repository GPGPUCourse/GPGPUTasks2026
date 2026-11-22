#ifndef sort_key_gpu_shared_pragma_once
#define sort_key_gpu_shared_pragma_once

#include "struct_helpers.h"

#if defined(__OPENCL_VERSION__) || defined(common_vk)
    #define GPUC_UINT uint
#else
    #include <stdint.h>
    #define GPUC_UINT uint32_t
#endif

GPU_STRUCT_BEGIN(SortKeyGPU)
    GPUC_UINT morton_code;
    GPUC_UINT triangle_index;
GPU_STRUCT_END(SortKeyGPU)

#if !defined(__OPENCL_VERSION__) && defined(__cplusplus)
    static_assert(sizeof(SortKeyGPU) == 8, "SortKeyGPU size mismatch");
#endif

#endif
