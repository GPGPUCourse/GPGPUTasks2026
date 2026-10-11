#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define MATRIX_WMMA_BLOCK_M 128
#define MATRIX_WMMA_BLOCK_N 128
#define MATRIX_WMMA_BLOCK_K 32
#define MATRIX_WMMA_WARPS 4
#define MATRIX_WMMA_PACK_THREADS 256

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
