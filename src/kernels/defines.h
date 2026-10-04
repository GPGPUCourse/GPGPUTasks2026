#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define TRANSPOSE_TILE_SIZE 32
#define TRANSPOSE_BLOCK_ROWS 8
#define MULTIPLY_OUTPUTS_PER_THREAD 2

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
