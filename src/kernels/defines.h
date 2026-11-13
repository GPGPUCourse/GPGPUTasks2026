#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define AO_SAMPLES   8
#define MAX_KERNEL_SIZE 8
#define DEPTH_EPS 0.2f
#define NORMAL_EPS 0.9f

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
