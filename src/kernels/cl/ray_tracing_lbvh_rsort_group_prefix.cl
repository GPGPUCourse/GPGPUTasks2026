#define RADIX_SIZE 16
#define RADIX_GROUPS_PER_TILE 256

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void ray_tracing_lbvh_rsort_group_prefix(
    __global const uint* hist,
    __global uint* groupLocalOffsets,
    __global uint* tileTotals,
    uint groupCount
) {
    __local uint scan[RADIX_GROUPS_PER_TILE];

    const uint lid = get_local_id(0);
    const uint tileAndDigit = get_group_id(0);
    const uint digit = tileAndDigit % RADIX_SIZE;
    const uint tile = tileAndDigit / RADIX_SIZE;
    const uint group = tile * RADIX_GROUPS_PER_TILE + lid;

    const uint count = group < groupCount
        ? hist[group * RADIX_SIZE + digit]
        : 0u;
    scan[lid] = count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < RADIX_GROUPS_PER_TILE; offset <<= 1) {
        const uint current = scan[lid];
        const uint previous = lid >= offset ? scan[lid - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan[lid] = current + previous;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint tileTotal = scan[RADIX_GROUPS_PER_TILE - 1];

    if (group < groupCount)
        groupLocalOffsets[group * RADIX_SIZE + digit] = scan[lid] - count;

    if (lid == 0)
        tileTotals[tile * RADIX_SIZE + digit] = tileTotal;
}
