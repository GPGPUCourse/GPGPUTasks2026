#define RADIX_BITS 4
#define RADIX_SIZE 16

__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void ray_tracing_lbvh_rsort_prefix(
    __global const uint* hist,
    __global uint* groupOffsets,
    uint groupCount
) {
    __local uint bucketTotals[RADIX_SIZE];
    // Запускаем ровно 16 work-items.
    const uint digit = get_local_id(0);

    // ==========================================
    // 1. Считаем общее количество элементов
    //    каждого bucket во всех work-group.
    // ==========================================

    uint total = 0;

    for (uint g = 0; g < groupCount; ++g) {
        total += hist[g * RADIX_SIZE + digit];
    }

    bucketTotals[digit] = total;

    barrier(CLK_LOCAL_MEM_FENCE);

    // ==========================================
    // 2. Находим начало текущего bucket.
    // ==========================================

    uint offset = 0;

    for (uint b = 0; b < digit; ++b) {
        offset += bucketTotals[b];
    }

    // ==========================================
    // 3. Находим offsets каждой work-group.
    // ==========================================

    for (uint g = 0; g < groupCount; ++g) {

        const uint index = g * RADIX_SIZE + digit;

        groupOffsets[index] = offset;

        offset += hist[index];
    }
}