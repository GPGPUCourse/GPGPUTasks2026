#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* a,
    __global const uint* cnt,
    __global const uint* pref,
    __global       uint* b,
    unsigned int n,
    unsigned int bit)
{
    int i = get_global_id(0);
    int li = get_local_id(0);
    int g = get_group_id(0);
    int ng = get_num_groups(0);

    __local uint vals[GROUP_SIZE];
    __local uint digs[GROUP_SIZE];
    __local uint offs[GROUP_SIZE];
    __local uint run[RADIX];

    int start = g * GROUP_SIZE;
    int ok = 0;
    if (start < n){
        ok = min((int)GROUP_SIZE, (int)(n - start));
    }

    if (li < ok) {
        vals[li] = a[start + li];
        digs[li] = (vals[li] >> bit) & (RADIX - 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (li == 0) {
        for (int d = 0; d < RADIX; d++) {
            run[d] = 0;
        }
        for (int t = 0; t < ok; t++) {
            int d = digs[t];
            offs[t] = run[d];
            run[d]++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < ok) {
        int d = digs[li];
        int pos = pref[d * ng + g] - cnt[d * ng + g] + offs[li];
        b[pos] = vals[li];
    }
}
