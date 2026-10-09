#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

// FP32 matmul configuration for Tesla V100 (sm_70).
#define CUDA_MM_BLOCK_M 128
#define CUDA_MM_BLOCK_N 128
#define CUDA_MM_BLOCK_K 16
#define CUDA_MM_THREADS 256

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
