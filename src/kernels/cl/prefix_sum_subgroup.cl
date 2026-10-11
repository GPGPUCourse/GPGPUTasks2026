#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define SCAN_USE_SUBGROUPS 1
#define SCAN_KERNEL_NAME prefix_sum_subgroup
#include "prefix_sum_01_reduction.cl"
