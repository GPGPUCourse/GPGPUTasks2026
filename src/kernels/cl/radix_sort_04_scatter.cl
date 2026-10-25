#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefixes,
    __global uint* output,
    unsigned int n,
    unsigned int shift)
{
    const uint lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t groups = get_num_groups(0);
    const size_t base = group * RADIX_BLOCK_SIZE;

    const uint chunks = RADIX_BLOCK_SIZE / 32;
    __local uint masks[RADIX_BUCKETS * (RADIX_BLOCK_SIZE / 32)];
    __local uint chunk_prefixes[RADIX_BUCKETS * (RADIX_BLOCK_SIZE / 32)];
    __local uint starts[RADIX_BUCKETS];
    __local uint ordered[RADIX_BLOCK_SIZE];
    uint values[RADIX_BLOCK_SIZE / GROUP_SIZE];

    for (uint i = lid; i < RADIX_BUCKETS * chunks; i += GROUP_SIZE) {
        masks[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const uint position = k * GROUP_SIZE + lid;
        if (base + position < n) {
            values[k] = input[base + position];
            const uint digit = (values[k] >> shift) & (RADIX_BUCKETS - 1);
            atomic_or(&masks[digit * chunks + position / 32], 1u << (position % 32));
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BUCKETS) {
        uint sum = 0;
        for (uint chunk = 0; chunk < chunks; ++chunk) {
            chunk_prefixes[lid * chunks + chunk] = sum;
            sum += popcount(masks[lid * chunks + chunk]);
        }
        starts[lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        uint sum = 0;
        for (uint digit = 0; digit < RADIX_BUCKETS; ++digit) {
            const uint count = starts[digit];
            starts[digit] = sum;
            sum += count;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const uint position = k * GROUP_SIZE + lid;
        if (base + position < n) {
            const uint digit = (values[k] >> shift) & (RADIX_BUCKETS - 1);
            const uint chunk = digit * chunks + position / 32;
            const uint earlier = (1u << (position % 32)) - 1;
            const uint rank = chunk_prefixes[chunk] + popcount(masks[chunk] & earlier);
            ordered[starts[digit] + rank] = values[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint k = 0; k < RADIX_BLOCK_SIZE / GROUP_SIZE; ++k) {
        const uint position = k * GROUP_SIZE + lid;
        if (base + position < n) {
            const uint value = ordered[position];
            const uint digit = (value >> shift) & (RADIX_BUCKETS - 1);
            const size_t destination = prefixes[digit * groups + group] + position - starts[digit];
            output[destination] = value;
        }
    }
}
