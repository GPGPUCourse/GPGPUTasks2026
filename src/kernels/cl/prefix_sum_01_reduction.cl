#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

void build_table(__local uint* qual)
{
    const unsigned int t = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int count = GROUP_SIZE / 2; count >= 1; count /= 2) {
        if (t < count) {
            const unsigned int v = count - 1 + t;
            qual[v] = qual[2 * v + 1] + qual[2 * v + 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

unsigned int prefix_by_mask(__local const uint* qual, unsigned int leaf, bool inclusive)
{
    unsigned int sum = inclusive ? qual[GROUP_SIZE - 1 + leaf] : 0;
    for (unsigned int h = 0; (GROUP_SIZE >> h) > 1; ++h) {
        if ((leaf >> h) & 1) {
            const unsigned int node = (GROUP_SIZE >> h) - 1 + (leaf >> h);
            sum += qual[node - 1];
        }
    }
    return sum;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* b,
    __global       uint* next_b,
    __global       uint* a,
    unsigned int m,
    unsigned int n,
    unsigned int level)
{
    __local uint qual[2 * GROUP_SIZE];

    const unsigned int t = get_local_id(0);
    const unsigned int k = get_global_id(0);
    const unsigned int shift = 8 * level;
    const unsigned int group_start = get_group_id(0) * GROUP_SIZE;
    const unsigned int j0 = group_start >> shift;
    const unsigned int table = j0 / GROUP_SIZE;

    const unsigned int j = table * GROUP_SIZE + t;
    qual[GROUP_SIZE - 1 + t] = (j < m) ? b[j] : 0;
    build_table(qual);

    if (t == 0 && group_start % ((ulong) GROUP_SIZE << shift) == 0)
        next_b[table] = qual[0];

    if (k < n && (level == 0 || (k >> shift) % GROUP_SIZE != 0)) {
        const unsigned int sum = prefix_by_mask(qual, (k >> shift) % GROUP_SIZE, level == 0);
        a[k] = (level == 0) ? sum : a[k] + sum;
    }
}
