#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#ifndef SCAN_USE_SUBGROUPS
#define SCAN_USE_SUBGROUPS 0
#endif

#ifndef SCAN_KERNEL_NAME
#define SCAN_KERNEL_NAME prefix_sum_01_reduction
#endif

#define TILE_INVALID         0u
#define TILE_AGGREGATE_READY 1u
#define TILE_PREFIX_READY    2u

#define SCAN_LOCAL_BANKS 32
#define SCAN_LOCAL_SIZE (SCAN_TILE_SIZE + SCAN_TILE_SIZE / SCAN_LOCAL_BANKS)
#define SCAN_LOCAL_INDEX(index) ((index) + (index) / SCAN_LOCAL_BANKS)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void SCAN_KERNEL_NAME(
    __global const uint* input,
    __global       uint* output,
    volatile __global ulong* tile_state,
    volatile __global uint* next_tile,
    unsigned int n,
    unsigned int tile_count)
{
    const unsigned int lane = get_local_id(0);

    __local uint exchange[SCAN_LOCAL_SIZE];
    __local uint claimed_tile;
    __local uint tile_offset;

    while (1) {
        if (lane == 0)
            claimed_tile = atomic_inc(next_tile);
        barrier(CLK_LOCAL_MEM_FENCE);

        const unsigned int tile = claimed_tile;
        if (tile >= tile_count)
            return;

        const unsigned int tile_begin = tile * SCAN_TILE_SIZE;

        #pragma unroll
        for (unsigned int i = 0; i < SCAN_VALUES_PER_THREAD; ++i) {
            const unsigned int index = tile_begin + i * GROUP_SIZE + lane;
            const unsigned int local_index = i * GROUP_SIZE + lane;
            exchange[SCAN_LOCAL_INDEX(local_index)] = index < n ? input[index] : 0u;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        uint values[SCAN_VALUES_PER_THREAD];
        #pragma unroll
        for (unsigned int i = 0; i < SCAN_VALUES_PER_THREAD; ++i) {
            const unsigned int local_index = lane * SCAN_VALUES_PER_THREAD + i;
            values[i] = exchange[SCAN_LOCAL_INDEX(local_index)];
        }

        #pragma unroll
        for (unsigned int i = 1; i < SCAN_VALUES_PER_THREAD; ++i)
            values[i] += values[i - 1];

        const uint thread_sum = values[SCAN_VALUES_PER_THREAD - 1];
        barrier(CLK_LOCAL_MEM_FENCE);

#if SCAN_USE_SUBGROUPS
        uint thread_inclusive = sub_group_scan_inclusive_add(thread_sum);
        const unsigned int subgroup_lane = get_sub_group_local_id();
        const unsigned int subgroup_id = get_sub_group_id();
        const unsigned int subgroup_size = get_sub_group_size();
        const unsigned int subgroup_count = get_num_sub_groups();

        if (subgroup_lane + 1 == subgroup_size)
            exchange[subgroup_id] = thread_inclusive;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            for (unsigned int i = 1; i < subgroup_count; ++i)
                exchange[i] += exchange[i - 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (subgroup_id != 0)
            thread_inclusive += exchange[subgroup_id - 1];
#else
        unsigned int src = 0;
        unsigned int dst = GROUP_SIZE;
        exchange[src + lane] = thread_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (unsigned int offset = 1; offset < GROUP_SIZE; offset <<= 1) {
            uint sum = exchange[src + lane];
            if (lane >= offset)
                sum += exchange[src + lane - offset];
            exchange[dst + lane] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);

            const unsigned int old_src = src;
            src = dst;
            dst = old_src;
        }

        const uint thread_inclusive = exchange[src + lane];
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint thread_offset = thread_inclusive - thread_sum;
        if (lane + 1 == GROUP_SIZE)
            exchange[0] = thread_inclusive;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lane == 0) {
            const uint aggregate = exchange[0];
            uint prefix_before_tile = 0;

            if (tile == 0) {
                atom_xchg(tile_state + tile,
                          ((ulong)TILE_PREFIX_READY << 32) | (ulong)aggregate);
            } else {
                atom_xchg(tile_state + tile,
                          ((ulong)TILE_AGGREGATE_READY << 32) | (ulong)aggregate);

                uint accumulated = 0;
                unsigned int predecessor = tile - 1;
                while (1) {
                    ulong state;
                    do {
                        state = atom_add(tile_state + predecessor, 0ul);
                    } while (state == TILE_INVALID);

                    const uint status = (uint)(state >> 32);
                    const uint value = (uint)state;
                    if (status == TILE_PREFIX_READY) {
                        prefix_before_tile = accumulated + value;
                        break;
                    }

                    accumulated += value;
                    --predecessor;
                }

                atom_xchg(tile_state + tile,
                          ((ulong)TILE_PREFIX_READY << 32) |
                          (ulong)(prefix_before_tile + aggregate));
            }

            tile_offset = prefix_before_tile;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const uint block_offset = tile_offset + thread_offset;
        #pragma unroll
        for (unsigned int i = 0; i < SCAN_VALUES_PER_THREAD; ++i)
            values[i] += block_offset;

        #pragma unroll
        for (unsigned int i = 0; i < SCAN_VALUES_PER_THREAD; ++i) {
            const unsigned int local_index = lane * SCAN_VALUES_PER_THREAD + i;
            exchange[SCAN_LOCAL_INDEX(local_index)] = values[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (unsigned int i = 0; i < SCAN_VALUES_PER_THREAD; ++i) {
            const unsigned int index = tile_begin + i * GROUP_SIZE + lane;
            const unsigned int local_index = i * GROUP_SIZE + lane;
            if (index < n)
                output[index] = exchange[SCAN_LOCAL_INDEX(local_index)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
