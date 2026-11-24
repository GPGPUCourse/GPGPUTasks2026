#define RADIX_SIZE 16

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void ray_tracing_lbvh_rsort_prefix(
    __global const uint* tileTotals,
    __global uint* tileOffsets,
    uint tileCount
) {
    __local uint bucketTotals[RADIX_SIZE];
    const uint digit = get_local_id(0);

    uint total = 0;
    for (uint tile = 0; tile < tileCount; ++tile)
        total += tileTotals[tile * RADIX_SIZE + digit];

    bucketTotals[digit] = total;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint offset = 0;
    for (uint b = 0; b < digit; ++b)
        offset += bucketTotals[b];

    for (uint tile = 0; tile < tileCount; ++tile) {
        const uint index = tile * RADIX_SIZE + digit;
        tileOffsets[index] = offset;
        offset += tileTotals[index];
    }
}