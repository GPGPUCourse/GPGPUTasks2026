#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X_TRANSPOSE 32
#define GROUP_SIZE_Y_TRANSPOSE 8
#define TILE_SIZE 16
#define GROUP_SIZE_X_MULTIPLY TILE_SIZE
#define GROUP_SIZE_Y_MULTIPLY TILE_SIZE

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
