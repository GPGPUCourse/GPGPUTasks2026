#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k, // 1,2,4,8,16. chunk_size=2,4,8,16,32
                   int  n)
{
    int i = get_global_id(0);
    if (i >= n) return;

    int chunk_start = (i / (2 * sorted_k)) * (2 * sorted_k);
    int chunk_len = min(2 * sorted_k, n - chunk_start);

    int a_start = chunk_start;
    int a_end = chunk_start + min(sorted_k, chunk_len);
    int b_start = a_end;
    int b_end = chunk_start + chunk_len;

    uint val = input_data[i];
    int is_a = (i < a_end);

    int subchunk_pos = is_a ? (i - a_start) : (i - b_start);

    int s_lo = is_a ? b_start : a_start;
    int s_hi = is_a ? b_end   : a_end;

    unsigned long threshold = (unsigned long)val + (unsigned long)(!is_a);

    int lo = s_lo;
    int hi = s_hi;
    while (lo < hi)
    {
        const int mid = lo + (hi - lo) / 2;
        if ((unsigned long)input_data[mid] < threshold)
            lo = mid + 1;
        else
            hi = mid;
    }

    int count = lo - s_lo;
    output_data[chunk_start + subchunk_pos + count] = val;
}
