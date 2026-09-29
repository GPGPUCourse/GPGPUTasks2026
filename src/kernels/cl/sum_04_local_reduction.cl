#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

// Та же заливка, что в sum_03. Атомиков нет: мастер пишет одну частичную сумму.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* restrict a,
                                     __global       uint* restrict b,
                                            unsigned int  n)
{
    const uint lid = get_local_id(0);
    const uint n4 = n >> 2;
    const uint base = (get_group_id(0) * 4u) * (uint)GROUP_SIZE + lid;
    __global const uint4* a4 = (__global const uint4*)a;

    uint4 v0 = (uint4)(0u);
    uint4 v1 = (uint4)(0u);
    uint4 v2 = (uint4)(0u);
    uint4 v3 = (uint4)(0u);
    if (base < n4)
        v0 = a4[base];
    if (base + (uint)GROUP_SIZE < n4)
        v1 = a4[base + (uint)GROUP_SIZE];
    if (base + 2u * (uint)GROUP_SIZE < n4)
        v2 = a4[base + 2u * (uint)GROUP_SIZE];
    if (base + 3u * (uint)GROUP_SIZE < n4)
        v3 = a4[base + 3u * (uint)GROUP_SIZE];

    const uint4 acc = (v0 + v1) + (v2 + v3);
    uint value = (acc.x + acc.y) + (acc.z + acc.w);

    const uint tail = n4 << 2;
    const uint stride = get_global_size(0);
    for (uint j = tail + get_global_id(0); j < n; j += stride)
        value += a[j];

    __local uint local_data[GROUP_SIZE];
    local_data[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (uint offset = (uint)GROUP_SIZE / 2u; offset > 0u; offset >>= 1) {
        if (lid < offset)
            local_data[lid] += local_data[lid + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u)
        b[get_group_id(0)] = local_data[0];
}
