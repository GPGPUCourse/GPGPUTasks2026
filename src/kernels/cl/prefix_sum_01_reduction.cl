#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global uint* pow2_sum,
    unsigned int start,
    unsigned int end)
{
    __local unsigned int buf[GROUP_SIZE * 4];
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int num = end - start;

    buf[2*lid] = (2*gid < num) ? pow2_sum[start + 2*gid] : 0;
    buf[2*lid + 1] = (2*gid + 1 < num) ? pow2_sum[start + 2*gid + 1] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int from = 0, to = GROUP_SIZE * 2, sz = to - from;
    unsigned int out_size = (num + 1) / 2;
    unsigned int base = end;
    unsigned int stride = GROUP_SIZE;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        unsigned int new_sz = (sz + 1) / 2;
        if (lid < new_sz) {
            unsigned int i = from + 2 * lid;
            unsigned int v = buf[i] + buf[i + 1];
            buf[to + lid] = v;
            unsigned int idx = stride * get_group_id(0) + lid;
            if (idx < out_size) {
                pow2_sum[base + idx] = v;
            }
        }
        sz = new_sz; from = to; to = from + sz;
        base += out_size; out_size = (out_size + 1) / 2;
        stride >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
