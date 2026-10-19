#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input,
    __global const uint* oracle,
    volatile __global uint* block_prefix,
    __global uint* prefix_sum_accum,
    uint n,
    uint leaf_count)
{
    uint local_id = get_local_id(0);
    uint block = get_group_id(0);
    size_t index = get_global_id(0);
    __local uint tree[GROUP_SIZE];
    __local uint block_offset;
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
        uint offset = 0;
        uint cursor = block;
        while (cursor != 0) {
            uint low = atomic_add(block_prefix + 2 * (cursor - 1), 0u);
            uint high = atomic_add(block_prefix + 2 * (cursor - 1) + 1, 0u);
            if (low != 0 && high != 0) {
                offset += (low - 1) | ((high - 1) << 16);
                break;
            }
            uint width = cursor & (0u - cursor);
            offset += oracle[(leaf_count + cursor) / width - 1];
            cursor -= width;
        }
        block_offset = offset;
        uint prefix = offset + tree[GROUP_SIZE - 1];
        atomic_xchg(block_prefix + 2 * block, (prefix & 0xffffu) + 1);
        atomic_xchg(block_prefix + 2 * block + 1, (prefix >> 16) + 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sum = block_offset;
    for (uint count = local_id + 1; count != 0; count &= count - 1) {
        sum += tree[count - 1];
    }
    if (index < n) {
        prefix_sum_accum[index] = sum;
    }
}
