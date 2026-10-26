#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* local_prefixes,
    __global const uint* block_prefixes,
    __global uint* output,
    unsigned int n,
    unsigned int bit)
{
    const uint tid = get_local_id(0);
    const uint block = get_group_id(0);
    const uint base = block * GROUP_SIZE;
    const uint i = base + tid;
    const uint count = min((uint)GROUP_SIZE, n - base);
    const uint blocks = (n - 1) / GROUP_SIZE + 1;
    const uint zeros_before = block == 0 ? 0 : block_prefixes[block - 1];
    const uint total_zeros = block_prefixes[blocks - 1];
    const uint local_zeros = local_prefixes[base + count - 1];
    __local uint ordered[GROUP_SIZE];

    if (tid < count) {
        const uint value = input[i];
        const uint zeros = local_prefixes[i];
        const uint one = (value >> bit) & 1u;
        const uint position = one ? local_zeros + tid - zeros : zeros - 1;
        ordered[position] = value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < count) {
        const uint position = tid < local_zeros
            ? zeros_before + tid
            : total_zeros + base - zeros_before + tid - local_zeros;
        output[position] = ordered[tid];
    }
}
