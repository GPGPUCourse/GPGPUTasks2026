#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

// Та же плиточная заливка, что в sum_03. Атомиков нет: мастер пишет сумму группы,
// хост повторяет запуск, пока частичных сумм не останется одна.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* restrict a,
                                     __global       uint* restrict b,
                                            unsigned int  n)
{
    const uint lid = get_local_id(0);
    const uint n4 = n >> 2;
    const uint tileSize = 4u * (uint)GROUP_SIZE;
    const uint tileStride = get_num_groups(0);

    uint4 acc = (uint4)(0u);
    for (uint tile = get_group_id(0); tile * tileSize < n4; tile += tileStride) {
        const uint tileBegin = tile * tileSize;
        const uint base = tileBegin + lid;
        if (tileBegin + tileSize <= n4) {
            const uint4 v0 = vload4(base, a);
            const uint4 v1 = vload4(base + (uint)GROUP_SIZE, a);
            const uint4 v2 = vload4(base + 2u * (uint)GROUP_SIZE, a);
            const uint4 v3 = vload4(base + 3u * (uint)GROUP_SIZE, a);
            acc += (v0 + v1) + (v2 + v3);
        } else {
            uint4 v0 = (uint4)(0u);
            uint4 v1 = (uint4)(0u);
            uint4 v2 = (uint4)(0u);
            uint4 v3 = (uint4)(0u);
            if (base < n4)
                v0 = vload4(base, a);
            if (base + (uint)GROUP_SIZE < n4)
                v1 = vload4(base + (uint)GROUP_SIZE, a);
            if (base + 2u * (uint)GROUP_SIZE < n4)
                v2 = vload4(base + 2u * (uint)GROUP_SIZE, a);
            if (base + 3u * (uint)GROUP_SIZE < n4)
                v3 = vload4(base + 3u * (uint)GROUP_SIZE, a);
            acc += (v0 + v1) + (v2 + v3);
        }
    }

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
