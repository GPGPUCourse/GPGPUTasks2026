#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define MATMUL_DIM 16

#define WARPS_PER_WG 8
#define WGSIZE_X 32
#define WGSIZE_Y 8
#define TILE_SIZE 16
#define TILES_X 8
#define TILES_Y 8

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
