#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

static uint count_less(__global const uint* data, uint len, uint x, bool with_equal)
{
    uint lo = 0, hi = len;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (data[mid] < x || (with_equal && data[mid] == x)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static uint count_less_local(__local const uint* data, uint len, uint x, bool with_equal)
{
    uint lo = 0, hi = len;
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (data[mid] < x || (with_equal && data[mid] == x)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort_local(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n)
{
    __local uint buf[GROUP_SIZE];

    const uint lid = get_local_id(0);
    const uint gid = get_global_id(0);

    buf[lid] = (gid < (uint) n) ? input_data[gid] : UINT_MAX;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 1; k < GROUP_SIZE; k *= 2) {
        uint v = buf[lid];
        uint start = (lid / (2 * k)) * (2 * k);
        uint pos;
        if (lid < start + k) {
            pos = (lid - start) + count_less_local(buf + start + k, k, v, false);
        } else {
            pos = (lid - start - k) + count_less_local(buf + start, k, v, true);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buf[start + pos] = v;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < (uint) n) {
        output_data[gid] = buf[lid];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint i = get_global_id(0);
    if (i >= (uint) n)
        return;

    const uint k = (uint) sorted_k;
    const uint v = input_data[i];

    const uint start = (i / (2 * k)) * (2 * k);
    const uint mid = min(start + k, (uint) n);
    const uint end = min(start + 2 * k, (uint) n);

    uint pos;
    if (i < mid) {
        pos = (i - start) + count_less(input_data + mid, end - mid, v, false);
    } else {
        pos = (i - mid) + count_less(input_data + start, mid - start, v, true);
    }
    output_data[start + pos] = v;
}
