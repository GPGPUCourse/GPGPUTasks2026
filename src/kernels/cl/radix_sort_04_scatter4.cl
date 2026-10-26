#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

void build_table(__local ulong2* qual)
{
    const ulong t = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (ulong count = GROUP_SIZE / 2; count >= 1; count /= 2) {
        if (t < count) {
            const ulong v = count - 1 + t;
            qual[v] = qual[2 * v + 1] + qual[2 * v + 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

ulong2 prefix_by_mask(__local const ulong2* qual, unsigned int leaf)
{
    ulong2 sum = 0;
    for (unsigned int h = 0; (GROUP_SIZE >> h) > 1; ++h) {
        if ((leaf >> h) & 1) {
            const unsigned int node = (GROUP_SIZE >> h) - 1 + (leaf >> h);
            sum += qual[node - 1];
        }
    }
    return sum;
}



__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter4(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* a, // изначальный массив
    __global const uint* pref, // префикс по 4 символам
    __global       uint* b, // ответик
    unsigned int radix_bits,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int t = get_local_id(0);
    const unsigned int j = get_global_id(0);
    const unsigned int gn = get_num_groups(0);
    const unsigned int g = get_group_id(0);
    const uint MASK = (1 << 4);

    __local ulong2 qual[2 * GROUP_SIZE];
    __local uint cnt[256];
    __local uint lans[GROUP_SIZE];
    
    const uint x = (j < n) ? a[j] : 0;
    const uint el = (x >> offset) & (MASK - 1);
    const uint valid = min((uint) GROUP_SIZE, n - g * GROUP_SIZE);

    qual[GROUP_SIZE - 1 + t] = (j < n) ? 
            (el < 8) ? (ulong2)(1UL << (8 * el), 0)
            : (ulong2)(0, 1UL << (8 * (el - 8)))
        : (ulong2)(0); // чего не сделаешь ради отсутствия бренчей

    if (t < MASK) {
        cnt[t] = pref[t * gn + g];
    }

    build_table(qual);
    

    const ulong2 total = qual[0];
    uint local_start[16];
    uint acc = 0;
    for (uint d = 0; d < 8; ++d) {
        local_start[d] = acc;
        acc += (total.x >> (8 * d)) & 0xFF;
    }
    for (uint d = 8; d < 16; ++d) {
        local_start[d] = acc;
        acc += (total.y >> (8 * (d - 8))) & 0xFF;
    }

    if (j < n) {
        ulong2 v = prefix_by_mask(qual, t);
        const uint rank = (el < 8 ? (v.x >> (8 * el)) : (v.y >> (8 * (el - 8)))) & 0xFF;
        lans[local_start[el] + rank] = x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    if (t < valid) {
        const uint y = lans[t];
        const uint d = (y >> offset) & (MASK - 1);
        b[cnt[d] + (t - local_start[d])] = y;
    }

}
