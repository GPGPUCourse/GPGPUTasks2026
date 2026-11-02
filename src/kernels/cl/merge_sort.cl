#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define MERGE_PATH(name, space) \
int name(space const uint* a, space const uint* b, int n_a, int n_b, int diag) { \
    int start = max(0, diag - n_b); \
    int end = min(diag, n_a); \
    while (start < end) { \
        int middle = (start + end) / 2; \
        if (a[middle] <= b[diag - middle - 1]) \
            start = middle + 1; \
        else \
            end = middle; \
    } \
    return start; \
}

MERGE_PATH(merge_path_left_global, __global)
MERGE_PATH(merge_path_left_local, __local)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint local_id = get_local_id(0);
    const uint group_offset = get_group_id(0) * GROUP_SIZE;
    const uint id = get_global_id(0);
    __local uint buffer[GROUP_SIZE];
    __local uint left_ids[2];

    if (group_offset >= n) return;

    int n_a, n_b, segment, diag;

    uint merged_len = 2 * sorted_k;

    __local uint temp[GROUP_SIZE];
    if (sorted_k == 1) {
        buffer[local_id] = (id < n) ? input_data[id] : 0xFFFFFFFFu;
        barrier(CLK_LOCAL_MEM_FENCE);

        __local uint* src = buffer;
        __local uint* dst = temp;

        for (int k = 1; k < GROUP_SIZE; k *= 2) {
            merged_len = 2 * k;

            segment = local_id / merged_len * merged_len;
            diag = local_id - segment;

            const int left = merge_path_left_local(
                src + segment,
                src + segment + k,
                k, k, diag
            );

            const int right = diag - left;
            const uint x = (left < k) ? src[segment + left] : 0xFFFFFFFFu;
            const uint y = (right < k) ? src[segment + k + right] : 0xFFFFFFFFu;

            dst[local_id] = min(x, y);
            barrier(CLK_LOCAL_MEM_FENCE);

            __local uint* ptr = src;
            src = dst;
            dst = ptr;
        }

        if (id < n)
            output_data[id] = src[local_id];
    }
    else {
        const int block_offset = group_offset / merged_len * merged_len;
        const int rest = n - block_offset;
        const int N_a = min(sorted_k, rest);
        const int N_b = min(sorted_k, rest - N_a);
        const int diag_low = group_offset - block_offset;
        const int diag_high = min(N_a + N_b, diag_low + GROUP_SIZE);
        if (local_id < 2)
            left_ids[local_id] = merge_path_left_global(
                input_data + block_offset,
                input_data + block_offset + N_a,
                N_a, N_b, local_id ? diag_high : diag_low
            );
        barrier(CLK_LOCAL_MEM_FENCE);

        const int left_start = left_ids[0];
        const int left_end = left_ids[1];
        const int right_start = diag_low - left_start;
        n_a = left_end - left_start;
        n_b = diag_high - left_end - right_start;

        if (local_id < n_a + n_b) {
            const int ptr = (local_id < n_a) ? block_offset + left_start + local_id : block_offset + sorted_k + right_start + local_id - n_a;
            buffer[local_id] = input_data[ptr];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        segment = 0;
        diag = local_id;

        if (id < n) {
            int left = merge_path_left_local(
                buffer + segment,
                buffer + segment + n_a,
                n_a, n_b, diag
            );
            int right = diag - left;
            uint x = (left < n_a) ? buffer[segment + left] : 0xFFFFFFFFu;
            uint y = (right < n_b) ? buffer[segment + n_a + right] : 0xFFFFFFFFu;
            output_data[id] = min(x, y);
        }
    }
}
