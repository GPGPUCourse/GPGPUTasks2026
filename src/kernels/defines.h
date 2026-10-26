#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#define RADIX_BITS     8
#define RADIX          (1 << RADIX_BITS)
#define RADIX_MASK     (RADIX - 1)
#define RADIX_NPASSES  (32 / RADIX_BITS)

#define SUB_BITS       1
#define SUB_RADIX      (1 << SUB_BITS)
#define SUB_MASK       (SUB_RADIX - 1)
#define SUB_NPASSES    (RADIX_BITS / SUB_BITS)

#define SORT_WG_SIZE   256
#define SORT_IPT       24
#define SORT_TILE      (SORT_WG_SIZE * SORT_IPT)

#define SCAN_WG_SIZE   256
#define SCAN_IPT       4
#define SCAN_TILE      (SCAN_WG_SIZE * SCAN_IPT)

#endif // pragma once
