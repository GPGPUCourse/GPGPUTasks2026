#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE 256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16
#define OFFSET(i) ((i >> 5) + (i >> 10))
#define PAD(i) i + OFFSET(i)
#define GS_LOG 8
#define PER_THREAD 4

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
