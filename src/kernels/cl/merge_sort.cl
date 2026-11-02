#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k,
                   uint  n)
{
    const uint i = get_global_id(0);

    if (i >= n) {
        return;
    }

    const uint width = (1u << sorted_k);
    const uint pair_id = i / (2 * width);
    const uint local_pos = i % (2 * width);

    const uint left_start = pair_id * (2 * width);
    const uint left_end = min(left_start + width, n);
    const uint right_start = left_end;
    const uint right_end = min(right_start + width, n);

    const uint left_size = left_end - left_start;
    const uint right_size = right_end - right_start;

    if (i < left_end) {
        const uint left_pos = left_start + local_pos;
        int l = -1;
        int r = right_size;
        while (r - l > 1) {
            const int mid = (r + l) / 2;
            if (input_data[right_start + mid] < input_data[left_pos]) {
                l = mid;
            } else {
                r = mid;
            }
        }
        output_data[left_pos + r] = input_data[left_pos];
    } else {
        const uint right_pos = left_start + local_pos;
        int l = -1;
        int r = left_size;
        while (r - l > 1) {
            const int mid = (r + l) / 2;
            if (input_data[left_start + mid] <= input_data[right_pos]) {
                l = mid;
            } else {
                r = mid;
            }
        }
        output_data[left_start + (local_pos - left_size) + r] = input_data[right_pos];
    }
}
