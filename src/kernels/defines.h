#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE 256

#define LOCAL_WORK_SIZE 512

#define COUNTING_THREAD_WORK 8
#define COUNTING_GROUP_SIZE (LOCAL_WORK_SIZE / COUNTING_THREAD_WORK)

// Настройки префиксных сумм
#define TILE 512
#define VECTOR_SIZE 16

#define SCATTER_THREAD_WORK 8
#define SCATTER_GROUP_SIZE (LOCAL_WORK_SIZE / SCATTER_THREAD_WORK)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
