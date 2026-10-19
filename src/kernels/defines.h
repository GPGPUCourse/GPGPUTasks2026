#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define SCAN_WARP_SIZE 32
#define SCAN_WARP_COUNT (GROUP_SIZE / SCAN_WARP_SIZE)
#define SCAN_ITEMS_PER_WORK_ITEM 4
#define SCAN_BLOCK_SIZE (GROUP_SIZE * SCAN_ITEMS_PER_WORK_ITEM)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
