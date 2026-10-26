#ifndef my_defines_vk // pragma once
#define my_defines_vk


#define RADIX_BITS 4

#define BUCKET_COUNT (1u << RADIX_BITS)
#define COUNTER_REDUCTION_GROUP_SIZE 256u
#define UINT4_PER_THREAD_IN_REDUCTION 1u

#define SCAN_GROUP_SIZE 512u
#define PREFIX_SUM_GROPU_DATA_SIZE 1024u

#define SCATTER_GROUP_SIZE \
    (COUNTER_REDUCTION_GROUP_SIZE * UINT4_PER_THREAD_IN_REDUCTION * 4u)


#define SCATTER_GROUP_WARPS (SCATTER_GROUP_SIZE / 32u)


#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
