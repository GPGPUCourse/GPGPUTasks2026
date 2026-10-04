#ifndef my_defines_vk // pragma once
#define my_defines_vk

// Транспонирование
#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

// Умножение
#define MUL_GROUP_SIZE_X 16
#define MUL_GROUP_SIZE_Y 16
#define MUL_THREAD_Y 8

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
