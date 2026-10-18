#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE 256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

// Параметры алгоритма префиксной суммы (scan):
// один тайл из SCAN_TILE элементов сканируется независимо от остальных (одной рабочей группой)
#define SCAN_ELEMS_PER_THREAD 4
#define SCAN_TILE (GROUP_SIZE * SCAN_ELEMS_PER_THREAD) // 1024 элемента на рабочую группу

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
