#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    
    if (i >= n)
        return;

    const int block_size = (1 << sorted_k);
    
    const int x = input_data[i];

    const int num_heap = i / block_size;
    const int num_pair_heap = (num_heap % 2 == 0) ? num_heap + 1 : num_heap - 1;

    const bool flag = (num_heap % 2 == 0);

    int lb = num_heap * block_size;
    int rb = min((num_heap + 1) * block_size, n);

    int lbn = num_pair_heap * block_size;
    int rbn = min((num_pair_heap + 1) * block_size, n);

    if (lbn >= n) {
        output_data[i] = x;
        return;
    }

    int l = lbn - 1;
    int r = rbn;


    while (l < r - 1) {
        int m = (l + r) / 2;

        if (flag) {
            if (x <= input_data[m])
                r = m;
            else
                l = m;
        }
        else {
            if (x < input_data[m])
                r = m;
            else
                l = m;
        }
    }

    if (flag)
        output_data[(i - lb) + (l - lbn + 1) + lb] = input_data[i];
    else
        output_data[(i - lb) + (r - lbn) + lbn] = input_data[i];       
}
