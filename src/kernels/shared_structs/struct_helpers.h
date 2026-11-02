#ifndef struct_helpers_pragma_once // pragma once
#define struct_helpers_pragma_once

#if defined(common_vk)
    // Vulkan GLSL
    #define GPU_STRUCT_BEGIN(name) struct name {
    #define GPU_STRUCT_END(name)   };
#else
    // OpenCL / CUDA / C++
    #define GPU_STRUCT_BEGIN(name) typedef struct name {
    #define GPU_STRUCT_END(name)   } name;
#endif

#endif // pragma once