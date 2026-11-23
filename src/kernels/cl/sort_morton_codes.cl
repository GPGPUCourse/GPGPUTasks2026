#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define RADIX 16
#define PER_THREAD 4
#define TILE (PER_THREAD * GROUP_SIZE)
#define ROWS (RADIX / 2)
#define LANES (GROUP_SIZE / ROWS)

#define CNT(d, t) cnt[(d) / 2 * (GROUP_SIZE + LANES) + (t) + (t) / ROWS]
#define CNT_SHIFT(d) ((d) % 2 * 16)
#define UNPACK(x, d) (((x) >> CNT_SHIFT(d)) & 0xFFFF)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* counts,
    unsigned int n,
    unsigned int shift)
{
    __local uint hist[RADIX];
    unsigned int tid = get_local_id(0);
    unsigned int base = get_group_id(0) * TILE;
    unsigned int end = min(base + TILE, n);

    if (tid < RADIX) {
        hist[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = base + tid; i < end; i += GROUP_SIZE) {
        atomic_inc(&hist[(input[i] >> shift) & (RADIX - 1)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < RADIX) {
        counts[tid * get_num_groups(0) + get_group_id(0)] = hist[tid];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* input,
    __global       uint* prefix_sum,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint temp[2 * GROUP_SIZE];
    const uint tid = get_local_id(0);
    const uint first = get_group_id(0) * (2 * GROUP_SIZE) + tid;
    const uint second = first + GROUP_SIZE;
    const uint first_value = first < n ? input[first] : 0;
    const uint second_value = second < n ? input[second] : 0;
    temp[tid] = first_value;
    temp[tid + GROUP_SIZE] = second_value;

    uint offset = 1;
    for (uint d = GROUP_SIZE; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        block_sums[get_group_id(0)] = temp[2 * GROUP_SIZE - 1];
        temp[2 * GROUP_SIZE - 1] = 0;
    }

    for (uint d = 1; d < 2 * GROUP_SIZE; d <<= 1) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid < d) {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            uint left_sum = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += left_sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (first < n) {
        prefix_sum[first] = temp[tid] + first_value;
    }
    if (second < n) {
        prefix_sum[second] = temp[tid + GROUP_SIZE] + second_value;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* block_prefix_sum,
    __global       uint* prefix_sum,
    unsigned int n)
{
    const uint block = get_group_id(0);
    if (block == 0) {
        return;
    }

    const uint prefix = block_prefix_sum[block - 1];
    const uint first = block * (2 * GROUP_SIZE) + get_local_id(0);
    const uint second = first + GROUP_SIZE;
    if (first < n) {
        prefix_sum[first] += prefix;
    }
    if (second < n) {
        prefix_sum[second] += prefix;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* input_ids,
    __global const uint* prefix_counts,
    __global       uint* output,
    __global       uint* output_ids,
    unsigned int n,
    unsigned int shift)
{
    __local uint keys[TILE];
    __local uint ids[TILE];
    __local uint cnt[ROWS * (GROUP_SIZE + LANES)];
    __local uint sums[GROUP_SIZE];
    __local uint bin_offset[RADIX];

    unsigned int tid = get_local_id(0);
    unsigned int base = get_group_id(0) * TILE;
    unsigned int size = min((unsigned int) TILE, n - base);

    for (unsigned int i = tid; i < size; i += GROUP_SIZE) {
        keys[i] = input[base + i];
        ids[i] = input_ids[base + i];
    }
    for (unsigned int d = 0; d < RADIX; d += 2) {
        CNT(d, tid) = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint vals[PER_THREAD];
    uint face_ids[PER_THREAD];
    uint ranks[PER_THREAD];
    for (unsigned int j = 0; j < PER_THREAD; ++j) {
        unsigned int i = tid * PER_THREAD + j;
        if (i < size) {
            vals[j] = keys[i];
            face_ids[j] = ids[i];
            unsigned int d = (vals[j] >> shift) & (RADIX - 1);
            ranks[j] = UNPACK(CNT(d, tid), d);
            CNT(d, tid) += 1u << CNT_SHIFT(d);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int lane = tid % LANES;
    __local uint* part = &CNT(2 * (tid / LANES), lane * ROWS);
    uint sum = 0;
    for (unsigned int k = 0; k < ROWS; ++k) {
        sum += part[k];
    }
    sums[tid] = sum;
    for (unsigned int offset = 1; offset < LANES; offset <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        uint add = lane >= offset ? sums[tid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        sums[tid] += add;
    }
    uint acc = sums[tid] - sum;
    for (unsigned int k = 0; k < ROWS; ++k) {
        uint c = part[k];
        part[k] = acc;
        acc += c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < RADIX) {
        uint offset = 0;
        for (unsigned int d = 0; d < tid; ++d) {
            offset += UNPACK(sums[d / 2 * LANES + LANES - 1], d);
        }
        bin_offset[tid] = offset;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int j = 0; j < PER_THREAD; ++j) {
        if (tid * PER_THREAD + j < size) {
            unsigned int d = (vals[j] >> shift) & (RADIX - 1);
            unsigned int idx = bin_offset[d] + ranks[j] + UNPACK(CNT(d, tid), d);
            keys[idx] = vals[j];
            ids[idx] = face_ids[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = tid; i < size; i += GROUP_SIZE) {
        unsigned int d = (keys[i] >> shift) & (RADIX - 1);
        unsigned int idx = d * get_num_groups(0) + get_group_id(0);
        uint global_offset = idx > 0 ? prefix_counts[idx - 1] : 0;
        output[global_offset + i - bin_offset[d]] = keys[i];
        output_ids[global_offset + i - bin_offset[d]] = ids[i];
    }
}
