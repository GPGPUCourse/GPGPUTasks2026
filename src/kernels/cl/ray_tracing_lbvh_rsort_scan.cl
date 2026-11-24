#define RADIX_BITS 4
#define RADIX_SIZE 16

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void ray_tracing_lbvh_rsort_scan(
    __global const uint* keysIn,
    __global const uint* valuesIn,

    __global uint* keysOut,
    __global uint* valuesOut,

    __global const uint* groupLocalOffsets,
    __global const uint* tileOffsets,

    uint n,
    uint shift
) {
    __local uint16 localScan[256];
    const uint groupId = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint ls = get_local_size(0);

    const uint gid = groupId * ls + lid;

    const uint valid = gid < n;

    const uint key = valid ? keysIn[gid] : 0u;
    const uint value = valid ? valuesIn[gid] : 0u;

    const uint digit =
        (key >> shift) & (RADIX_SIZE - 1);

    // ==========================================
    // 1. Создаём one-hot вектор.
    // ==========================================

    uint16 flags = (uint16)(
        valid && digit == 0  ? 1u : 0u,
        valid && digit == 1  ? 1u : 0u,
        valid && digit == 2  ? 1u : 0u,
        valid && digit == 3  ? 1u : 0u,
        valid && digit == 4  ? 1u : 0u,
        valid && digit == 5  ? 1u : 0u,
        valid && digit == 6  ? 1u : 0u,
        valid && digit == 7  ? 1u : 0u,
        valid && digit == 8  ? 1u : 0u,
        valid && digit == 9  ? 1u : 0u,
        valid && digit == 10 ? 1u : 0u,
        valid && digit == 11 ? 1u : 0u,
        valid && digit == 12 ? 1u : 0u,
        valid && digit == 13 ? 1u : 0u,
        valid && digit == 14 ? 1u : 0u,
        valid && digit == 15 ? 1u : 0u
    );

    localScan[lid] = flags;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < ls; offset <<= 1) {
        const uint16 current = localScan[lid];
        const uint16 previous = lid >= offset
            ? localScan[lid - offset]
            : (uint16)(0u);

        barrier(CLK_LOCAL_MEM_FENCE);

        localScan[lid] = current + previous;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (valid) {
        const uint tileId = groupId / 256;
        const uint bucketOffset = groupLocalOffsets[groupId * RADIX_SIZE + digit]
                                + tileOffsets[tileId * RADIX_SIZE + digit];
        const uint localRank = localScan[lid][digit] - 1u;
        const uint destination = bucketOffset + localRank;

        keysOut[destination] = key;
        valuesOut[destination] = value;
    }
}