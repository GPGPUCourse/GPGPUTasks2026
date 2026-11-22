#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "../shared_structs/sort_key_gpu_shared.h"

void merge_sort_local(
    __local const SortKeyGPU*    input_data,
    __local SortKeyGPU*          output_data,
    uint                         thread_id,
    uint                         sorted_k,
    uint                         n
) {
    if (thread_id >= n) {
        return;
    }

    const uint size = sorted_k * 2;
    const uint start = (thread_id / size) * size;

    const uint l_start = start;
    const uint l_end = min(start + (uint)sorted_k, (uint)n);
    const uint r_start = l_end;
    const uint r_end = min(start + size, (uint)n);
    const uint is_left = l_start <= thread_id && thread_id < l_end;

    const SortKeyGPU elem = input_data[thread_id];
    int l = (int)(is_left ? r_start : l_start) - 1;
    int r = is_left ? r_end : l_end;
    while (r - l > 1) {
        const int m = l + (r - l) / 2;
        if (input_data[m].morton_code < elem.morton_code || (!is_left && input_data[m].morton_code == elem.morton_code)) {
            l = m;
        } else {
            r = m;
        }
    }

    const uint less_cnt = is_left ? r - r_start : r - l_start;
    const uint my_start = is_left ? l_start : r_start;
    const uint target_idx = start + (thread_id - my_start) + less_cnt;
    output_data[target_idx] = elem;
}

void merge_sort_global(
    __global const SortKeyGPU*    input_data,
    __global SortKeyGPU*          output_data,
    uint                          thread_id,
    uint                          sorted_k,
    uint                          n
) {
    if (thread_id >= n) {
        return;
    }

    const uint size = sorted_k * 2;
    const uint start = (thread_id / size) * size;

    const uint l_start = start;
    const uint l_end = min(start + (uint)sorted_k, (uint)n);
    const uint r_start = l_end;
    const uint r_end = min(start + size, (uint)n);
    const uint is_left = l_start <= thread_id && thread_id < l_end;

    const SortKeyGPU elem = input_data[thread_id];
    int l = (int)(is_left ? r_start : l_start) - 1;
    int r = is_left ? r_end : l_end;
    while (r - l > 1) {
        const int m = l + (r - l) / 2;
        if (input_data[m].morton_code < elem.morton_code || (!is_left && input_data[m].morton_code == elem.morton_code)) {
            l = m;
        } else {
            r = m;
        }
    }

    const uint less_cnt = is_left ? r - r_start : r - l_start;
    const uint my_start = is_left ? l_start : r_start;
    const uint target_idx = start + (thread_id - my_start) + less_cnt;
    output_data[target_idx] = elem;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const SortKeyGPU*    input_data,
    __global SortKeyGPU*          output_data,
    uint                          sorted_k,
    uint                          n
) {
    __local SortKeyGPU buf1[GROUP_SIZE];
    __local SortKeyGPU buf2[GROUP_SIZE];

    const uint thread_id = get_global_id(0);
    if (sorted_k < GROUP_SIZE) {
        const uint block_idx = get_local_id(0);
        const uint block_start = get_group_id(0) * GROUP_SIZE;
        const uint block_size = min(n - block_start, (uint)GROUP_SIZE);
        if (block_idx < block_size) {
            buf1[block_idx] = input_data[thread_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        bool is_first = true;
        while (sorted_k < GROUP_SIZE) {
            if (is_first) {
                merge_sort_local(buf1, buf2, block_idx, sorted_k, block_size);
            } else {
                merge_sort_local(buf2, buf1, block_idx, sorted_k, block_size);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            is_first = !is_first;
            sorted_k *= 2;
        }

        if (thread_id < n) {
            output_data[thread_id] = is_first ? buf1[block_idx] : buf2[block_idx];
        }
        return;
    }

    merge_sort_global(input_data, output_data, thread_id, sorted_k, n);
}
