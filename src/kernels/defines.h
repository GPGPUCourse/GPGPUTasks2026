#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define MERGE_GROUP_SIZE 256
#define MERGE_ITEMS_PER_THREAD 8
#define MERGE_BLOCK_SIZE (MERGE_GROUP_SIZE * MERGE_ITEMS_PER_THREAD)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
