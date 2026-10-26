#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif
#include "../defines.h"
#include "helpers/rassert.cl"
#include "helpers/exclusive_scan.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* locally_sorted,
    __global uint* histogram,
    uint n, uint shift, uint groups)
{
    const uint lane = get_local_id(0);
    const uint block = get_group_id(0);
    const size_t index = get_global_id(0);
    const uint count = min((uint)GROUP_SIZE, n - block * GROUP_SIZE);
    __local uint values[GROUP_SIZE];
    __local uint sums[GROUP_SIZE];
    values[lane] = index < n ? input[index] : 0xffffffffu;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint bit = shift; bit < shift + RADIX_BITS; ++bit) {
        uint value = values[lane];
        uint zero = ((value >> bit) & 1u) == 0;
        sums[lane] = zero;
        uint zeros = exclusive_scan(sums, lane);
        uint position = zero ? sums[lane] : zeros + lane - sums[lane];
        values[position] = value;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (index < n) locally_sorted[index] = values[lane];
    if (lane < RADIX_BUCKETS) {
        uint bounds[2];
        for (uint end = 0; end < 2; ++end) {
            uint lo = 0, hi = count;
            while (lo < hi) {
                uint mid = lo + (hi - lo) / 2;
                uint digit = (values[mid] >> shift) & (RADIX_BUCKETS - 1);
                if (digit < lane + end) lo = mid + 1;
                else hi = mid;
            }
            bounds[end] = lo;
        }
        histogram[lane * groups + block] = bounds[1] - bounds[0];
    }
}
