#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RADIX_BITS      8
#define RADIX           (1 << RADIX_BITS)               // число корзин (значений цифры) за проход
#define KEYS_PER_THREAD 32
#define TILE_SIZE       (GROUP_SIZE * KEYS_PER_THREAD)  // ключей на рабочую группу в кернелах 01 и 04

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
