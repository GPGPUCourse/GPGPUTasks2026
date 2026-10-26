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
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_counts,
    __global       uint* output,
    unsigned int n,
    unsigned int shift)
{
    __local uint keys[TILE];
    __local uint cnt[ROWS * (GROUP_SIZE + LANES)];
    __local uint sums[GROUP_SIZE];
    __local uint bin_offset[RADIX];

    unsigned int tid = get_local_id(0);
    unsigned int base = get_group_id(0) * TILE;
    unsigned int size = min((unsigned int) TILE, n - base);

    for (unsigned int i = tid; i < size; i += GROUP_SIZE) {
        keys[i] = input[base + i];
    }
    for (unsigned int d = 0; d < RADIX; d += 2) {
        CNT(d, tid) = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint vals[PER_THREAD];
    uint ranks[PER_THREAD];
    for (unsigned int j = 0; j < PER_THREAD; ++j) {
        unsigned int i = tid * PER_THREAD + j;
        if (i < size) {
            vals[j] = keys[i];
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
            keys[bin_offset[d] + ranks[j] + UNPACK(CNT(d, tid), d)] = vals[j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = tid; i < size; i += GROUP_SIZE) {
        unsigned int d = (keys[i] >> shift) & (RADIX - 1);
        unsigned int idx = d * get_num_groups(0) + get_group_id(0);
        uint global_offset = idx > 0 ? prefix_counts[idx - 1] : 0;
        output[global_offset + i - bin_offset[d]] = keys[i];
    }
}
