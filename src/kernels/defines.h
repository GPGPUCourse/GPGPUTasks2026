#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SHIFT  8
#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define ACC_SHIFT 8
#define GROUP_SIZE_ACC (1 << ACC_SHIFT)

#if 0
#  define WARP_SIZE 16
#  define WARP_SHIFT 4
#else
#  define WARP_SIZE 32
#  define WARP_SHIFT 5
#endif

#define USE_SUBGROUPS 0


#define RASSERT_ENABLED 1 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
