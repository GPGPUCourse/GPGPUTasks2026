#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

/*
Original Paper:
    https://dl.acm.org/doi/epdf/10.1145/3694906.3743326
Original Paper code:
    https://github.com/b0nes164/Decoupled-Fallback-Paper
Opencl adaptation:
    https://gist.github.com/mambastudio/32f15d82e84ade4d936dc91b40b7d9bd
I just fused projection on digit here
*/

#include "../defines.h"
#include "helpers/rassert.cl"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#define ITEMS_PER_THREAD 8u
#define STATE_NOT_READY 0x00000000u
#define STATE_READY 0x40000000u
#define STATE_INCLUSIVE 0x80000000u
#define VALUE_MASK 0x0000FFFFu
#define STATE_MASK 0xC0000000u
#define MAX_SPINS 8u

#define SPINE_RETRY 0u
#define SPINE_COMPLETE 1u
#define SPINE_PENDING 2u

#define CONTINUE 0u
#define STOP 1u
#define FALLBACK 2u

typedef struct {
    uint state;
    uint payload;
} SpineEntry32;

typedef struct {
    uint action;
    uint tile_id;
    uint prefix;
} LookbackResult32;

SpineEntry32 read_spine32(
    __global volatile uint* spine,
    uint tile_id,
    __local uint* scratch)
{
    const uint base = tile_id * 2u;
    const uint lid = get_local_id(0);
    uint spins = 0u;
    SpineEntry32 result = { STATE_NOT_READY, 0u };

    while (spins < MAX_SPINS) {
        if (lid < 2u)
            scratch[lid] = atom_add(&spine[base + lid], 0u);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            const uint word0 = scratch[0];
            const uint word1 = scratch[1];
            const uint state0 = word0 & STATE_MASK;
            const uint state1 = word1 & STATE_MASK;

            if (state0 == state1 && state0 != STATE_NOT_READY) {
                scratch[2] = SPINE_COMPLETE;
                scratch[3] = state0;
                scratch[4] = (word0 & VALUE_MASK) | ((word1 & VALUE_MASK) << 16u);
            } else if (state0 == STATE_INCLUSIVE || state1 == STATE_INCLUSIVE) {
                scratch[2] = SPINE_PENDING;
            } else {
                scratch[2] = SPINE_RETRY;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (scratch[2] == SPINE_COMPLETE) {
            result.state = scratch[3];
            result.payload = scratch[4];
            return result;
        }
        if (scratch[2] == SPINE_RETRY)
            ++spins;
    }

    return result;
}

SpineEntry32 write_spine32(
    __global volatile uint* spine,
    uint tile_id,
    SpineEntry32 entry,
    __local uint* scratch)
{
    const uint base = tile_id * 2u;
    const uint lid = get_local_id(0);

    if (lid < 2u) {
        const uint part = lid == 0u
            ? (entry.payload & VALUE_MASK)
            : ((entry.payload >> 16u) & VALUE_MASK);
        const uint word = entry.state | part;
        scratch[lid] = atom_max(&spine[base + lid], word);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0u) {
        const uint old_state0 = scratch[0] & STATE_MASK;
        const uint old_state1 = scratch[1] & STATE_MASK;

        if (old_state0 == STATE_INCLUSIVE && old_state1 == STATE_INCLUSIVE) {
            scratch[3] = STATE_INCLUSIVE;
            scratch[4] = (scratch[0] & VALUE_MASK) | ((scratch[1] & VALUE_MASK) << 16u);
        } else if (entry.state == STATE_READY && (old_state0 == STATE_INCLUSIVE || old_state1 == STATE_INCLUSIVE)) {
            scratch[3] = STATE_NOT_READY;
            scratch[4] = 0u;
        } else {
            scratch[3] = entry.state;
            scratch[4] = entry.payload;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    SpineEntry32 published;
    published.state = scratch[3];
    published.payload = scratch[4];
    return published;
}

LookbackResult32 lookback32(
    __global volatile uint* spine,
    uint previous_tile,
    __local uint* scratch)
{
    uint accumulated = 0u;
    uint tile_id = previous_tile;

    while (1) {
        const SpineEntry32 entry = read_spine32(spine, tile_id, scratch);

        if (entry.state == STATE_INCLUSIVE) {
            LookbackResult32 result = {
                STOP, tile_id, accumulated + entry.payload
            };
            return result;
        }
        if (entry.state == STATE_READY) {
            accumulated += entry.payload;
            --tile_id;
            continue;
        }

        LookbackResult32 result = { FALLBACK, tile_id, accumulated };
        return result;
    }
}

SpineEntry32 fallback_insert(
    __global volatile uint* spine,
    uint tile_id,
    uint reduction,
    __local uint* scratch)
{
    SpineEntry32 fallback;
    fallback.state = tile_id == 0u ? STATE_INCLUSIVE : STATE_READY;
    fallback.payload = reduction;

    SpineEntry32 published = write_spine32(spine, tile_id, fallback, scratch);
    while (published.state == STATE_NOT_READY)
        published = read_spine32(spine, tile_id, scratch);
    return published;
}

/* Each work-item owns eight adjacent values; return an exclusive local scan. */
uint8 register_blocked_scan8(
    __global const uint* input,
    uint length,
    uint digit,
    uint tile_id,
    __local uint* aux)
{
    const uint lid = get_local_id(0);
    const uint base = (tile_id * get_local_size(0) + lid) * ITEMS_PER_THREAD;

    // Fast vector load on full vectors; guard only the final partial vector.
    uint8 keys;
    if (base + 7u < length) {
        keys = vload8(0, input + base);
    } else {
        keys = (uint8)(base + 0u < length ? input[base + 0u] : 0u,
            base + 1u < length ? input[base + 1u] : 0u,
            base + 2u < length ? input[base + 2u] : 0u,
            base + 3u < length ? input[base + 3u] : 0u,
            base + 4u < length ? input[base + 4u] : 0u,
            base + 5u < length ? input[base + 5u] : 0u,
            base + 6u < length ? input[base + 6u] : 0u,
            base + 7u < length ? input[base + 7u] : 0u);
    }

    const uint8 projection = (uint8)((keys.s0 >> digit) & 1u,
        (keys.s1 >> digit) & 1u,
        (keys.s2 >> digit) & 1u,
        (keys.s3 >> digit) & 1u,
        (keys.s4 >> digit) & 1u,
        (keys.s5 >> digit) & 1u,
        (keys.s6 >> digit) & 1u,
        (keys.s7 >> digit) & 1u);
    uint8 values = projection;

    values.s1 += values.s0;
    values.s2 += values.s1;
    values.s3 += values.s2;
    values.s4 += values.s3;
    values.s5 += values.s4;
    values.s6 += values.s5;
    values.s7 += values.s6;

    aux[lid] = values.s7;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1u; offset < get_local_size(0); offset <<= 1u) {
        const uint preceding = lid >= offset ? aux[lid - offset] : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= offset)
            aux[lid] += preceding;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint preceding = lid == 0u ? 0u : aux[lid - 1u];
    return values + (uint8)(preceding)-projection; // -projection, so its exclusive
}

uint register_blocked_reduction8(
    __global const uint* input,
    uint length,
    uint digit,
    uint tile_id,
    __local uint* aux)
{
    (void)register_blocked_scan8(input, length, digit, tile_id, aux);
    return aux[get_local_size(0) - 1u];
}

/*
 * Single-pass digit-prefix scan. Global size must be num_tiles * GROUP_SIZE,
 * local size exactly GROUP_SIZE, and scan storage must cover the padded size.
 */
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global uint* scan,
    __global volatile uint* spine,
    __global volatile uint* tile_bump,
    __global uint* total,
    uint length,
    uint digit)
{
    const uint lid = get_local_id(0);
    __local uint my_tile;
    __local uint scan_scratch[GROUP_SIZE];
    __local uint spine_scratch[5];

    if (lid == 0u)
        my_tile = atom_inc(tile_bump);
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint tile_id = my_tile;
    const uint num_tiles = get_num_groups(0);

    const uint8 local_exclusive = register_blocked_scan8(input, length, digit, tile_id, scan_scratch);
    const uint tile_reduction = scan_scratch[GROUP_SIZE - 1u];

    SpineEntry32 local_entry;
    local_entry.state = tile_id == 0u ? STATE_INCLUSIVE : STATE_READY;
    local_entry.payload = tile_reduction;
    (void)write_spine32(spine, tile_id, local_entry, spine_scratch);

    uint prefix = 0u;

    if (tile_id != 0u) {
        uint previous_tile = tile_id - 1u;
        uint complete = 0u;

        while (!complete) {
            const LookbackResult32 lookback = lookback32(spine, previous_tile, spine_scratch);
            prefix += lookback.prefix;

            if (lookback.action == STOP) {
                complete = 1u;
            } else {
                const uint fallback_reduction = register_blocked_reduction8(
                    input, length, digit, lookback.tile_id, scan_scratch);
                const SpineEntry32 published = fallback_insert(
                    spine, lookback.tile_id, fallback_reduction, spine_scratch);
                prefix += published.payload;

                if (published.state == STATE_INCLUSIVE || lookback.tile_id == 0u) {
                    complete = 1u;
                } else {
                    previous_tile = lookback.tile_id - 1u;
                }
            }
        }

        SpineEntry32 inclusive_entry;
        inclusive_entry.state = STATE_INCLUSIVE;
        inclusive_entry.payload = prefix + tile_reduction;
        (void)write_spine32(spine, tile_id, inclusive_entry, spine_scratch);
    }

    if (lid == 0u && tile_id == num_tiles - 1u)
        total[0] = prefix + tile_reduction;

    const uint out_base = (tile_id * get_local_size(0) + lid) * ITEMS_PER_THREAD;
    vstore8(local_exclusive + (uint8)(prefix), 0, scan + out_base);
}
