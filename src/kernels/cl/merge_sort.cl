#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);

    if (i >= n || n <= 0 || sorted_k <= 0)
        return;

    const ulong N = (ulong)n;
    const ulong K = (ulong)sorted_k;
    const ulong g = (ulong)i;

    const ulong width = 2ul * K;
    const ulong base = (g / width) * width;

    ulong mid = base + K;
    if (mid > N)
        mid = N;

    ulong end = base + width;
    if (end > N)
        end = N;

    const ulong len_a = mid - base;
    const ulong len_b = end - mid;
    const ulong rank = g - base;

    if (len_b == 0) {
        output_data[i] = input_data[i];
        return;
    }

    ulong low = (rank > len_b) ? rank - len_b : 0;
    ulong high = (rank < len_a) ? rank : len_a;
    ulong q = 0;

    while (low <= high) {
        q = low + (high - low) / 2;

        const ulong j = rank - q;

        if (q > 0 && j < len_b &&
            input_data[(size_t)(base + q - 1)] >
            input_data[(size_t)(mid + j)]) {

            high = q - 1;
        }

        else if (j > 0 && q < len_a &&
                 input_data[(size_t)(mid + j - 1)] >=
                 input_data[(size_t)(base + q)]) {

            low = q + 1;
        }
        else {
            break;
        }
    }
 
    const ulong j = rank - q;
    uint result;

    if (q == len_a) {
        result = input_data[(size_t)(mid + j)];
    }
    else if (j == len_b) {
        result = input_data[(size_t)(base + q)];
    }
    else {
        const uint a = input_data[(size_t)(base + q)];
        const uint b = input_data[(size_t)(mid + j)];
        result = (a <= b) ? a : b;
    }

    output_data[i] = result;
}
