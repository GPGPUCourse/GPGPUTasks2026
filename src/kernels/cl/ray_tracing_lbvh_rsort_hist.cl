#define RADIX_BITS 4
#define RADIX_SIZE 16

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void ray_tracing_lbvh_rsort_hist(
    __global const uint* keys,
    __global uint* hist,

    uint n,
    uint shift
) {
    __local uint localHist[RADIX_SIZE];
    const uint lid = get_local_id(0);
    const uint ls = get_local_size(0);

    // Инициализация локальной гистограммы.
    if (lid < RADIX_SIZE)
        localHist[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint gid = get_group_id(0) * ls + lid;

    if (gid < n) {

        const uint digit =
            (keys[gid] >> shift) & (RADIX_SIZE - 1);

        atomic_inc(
            (volatile __local unsigned int*)&localHist[digit]
        );
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Сохраняем histogram этой work-group.
    if (lid < RADIX_SIZE) {
        hist[get_group_id(0) * RADIX_SIZE + lid] =
            localHist[lid];
    }
}