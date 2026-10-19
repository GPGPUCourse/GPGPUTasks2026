#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global uint* oracle,
    __global uint* block_prefix,
    uint n,
    uint leaf_count)
{
    uint local_id = get_local_id(0);
    uint block = get_group_id(0);
    size_t index = get_global_id(0);
    __local uint tree[GROUP_SIZE];
    tree[local_id] = index < n ? input[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        uint right = (local_id + 1) * (stride << 1) - 1;
        if (right < GROUP_SIZE) {
            tree[right] += tree[right - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        oracle[leaf_count + block] = tree[GROUP_SIZE - 1];
        uint block_count = n / GROUP_SIZE + (n % GROUP_SIZE != 0);
        if (block_count + block < leaf_count) {
            oracle[leaf_count + block_count + block] = 0;
        }
        block_prefix[2 * block] = 0;
        block_prefix[2 * block + 1] = 0;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_build_oracle(__global uint* oracle, uint level_size)
{
    uint index = get_global_id(0);
    if (index < level_size) {
        uint node = level_size + index;
        oracle[node] = oracle[2 * node] + oracle[2 * node + 1];
    }
}
