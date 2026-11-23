#include "../defines.h"

static inline uint lbvh_lower_bound(__global const uint* codes, uint lo, uint hi, uint value)
{
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (codes[mid] < value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline uint lbvh_upper_bound(__global const uint* codes, uint lo, uint hi, uint value)
{
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (codes[mid] <= value) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_merge_pairs(
    __global const uint* input_codes,
    __global const uint* input_ids,
    __global uint* output_codes,
    __global uint* output_ids,
    uint nfaces,
    uint run_length)
{
    const uint i = get_global_id(0);
    if (i >= nfaces) return;

    const uint pair_length = 2 * run_length;
    const uint first = (i / pair_length) * pair_length;
    const uint middle = min(first + run_length, nfaces);
    const uint last = min(first + pair_length, nfaces);
    const uint code = input_codes[i];
    uint destination;
    if (i < middle) {
        const uint before = lbvh_lower_bound(input_codes, middle, last, code) - middle;
        destination = i + before;
    } else {
        const uint before = lbvh_upper_bound(input_codes, first, middle, code) - first;
        destination = i + before - (middle - first);
    }
    output_codes[destination] = code;
    output_ids[destination] = input_ids[i];
}
