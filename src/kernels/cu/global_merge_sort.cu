#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

#include "pos_bin_search.cuh"

__device__ unsigned merge_path(
    const unsigned* num_buf,
    unsigned a_start,
    unsigned b_start,
    unsigned chunk_size,
    unsigned d,
    unsigned n)
{
    unsigned i_min, i_max, i_mean;
    if (d < chunk_size) {
        i_min = 0;
        i_max = d;
    } else {
        i_min = d - chunk_size;
        i_max = chunk_size;
    }
    i_mean = (i_min + i_max) / 2;

    unsigned l[2] = { i_min, d - i_min };
    unsigned r[2] = { i_max, d - i_max };
    unsigned p[2] = { i_mean, d - i_mean };

    while (l[0] != r[0]) {
        unsigned a_index = a_start + p[0];
        unsigned b_index = b_start + p[1] - 1;

        unsigned a = (a_index < n) ? num_buf[a_index] : ~0U;
        unsigned b = (b_index < n) ? num_buf[b_index] : ~0U;

        if (b >= a) {
            l[0] = p[0] + 1;
            l[1] = p[1] - 1;
        } else {
            r[0] = p[0];
            r[1] = p[1];
        }

        p[0] = (l[0] + r[0]) / 2;
        p[1] = d - p[0];
    }

    return p[0];
}

__global__ void global_merge_sort(
    const unsigned* num_buf,
    unsigned* sorted_buf,
    unsigned sorted_buf_size,
    unsigned n)
{
    unsigned loc_id = threadIdx.x;
    unsigned wg_id = blockIdx.x;
    unsigned glob_id = wg_id * GROUP_SIZE + loc_id;

    unsigned group_sorted_start = wg_id * GROUP_SIZE;

    unsigned pair_start = (glob_id / sorted_buf_size) * sorted_buf_size;
    unsigned chunk_size = sorted_buf_size / 2;
    unsigned a_start = pair_start;
    unsigned b_start = pair_start + chunk_size;

    unsigned d_start = group_sorted_start - pair_start;
    unsigned d_end = d_start + GROUP_SIZE;

    __shared__ unsigned i_start, i_end;
    __shared__ unsigned j_start, j_end;

    if (loc_id == 0) {
        i_start = merge_path(num_buf, a_start, b_start, chunk_size, d_start, n);
        j_start = d_start - i_start;

        i_end = merge_path(num_buf, a_start, b_start, chunk_size, d_end, n);
        j_end = d_end - i_end;
    }
    __syncthreads();

    unsigned a_count = i_end - i_start;
    unsigned b_count = j_end - j_start;

    __shared__ unsigned loc_num_buf[GROUP_SIZE];
    __shared__ unsigned loc_sorted_buf[GROUP_SIZE];
    if (loc_id < a_count) {
        unsigned a_load_idx = a_start + i_start + loc_id;
        loc_num_buf[loc_id] = (a_load_idx < n) ? num_buf[a_load_idx] : ~0U;
    } else {
        unsigned b_load_idx = b_start + j_start + (loc_id - a_count);
        loc_num_buf[loc_id] = (b_load_idx < n) ? num_buf[b_load_idx] : ~0U;
    }
    __syncthreads();

    unsigned pos_in_loc_chunk, pos_in_neigh_loc_chunk, pos_in_loc_sorted_buf;
    if (loc_id < a_count) {
        pos_in_loc_chunk = loc_id;
        pos_in_neigh_loc_chunk = pos_bin_search(
            loc_num_buf + a_count, loc_num_buf[loc_id], b_count, chunk_side::left);
    } else {
        pos_in_loc_chunk = loc_id - a_count;
        pos_in_neigh_loc_chunk = pos_bin_search(
            loc_num_buf, loc_num_buf[loc_id], a_count, chunk_side::right);
    }
    pos_in_loc_sorted_buf = pos_in_loc_chunk + pos_in_neigh_loc_chunk;
    loc_sorted_buf[pos_in_loc_sorted_buf] = loc_num_buf[loc_id];
    __syncthreads();

    if (glob_id < n)
        sorted_buf[glob_id] = loc_sorted_buf[loc_id];
}

namespace cuda {
void global_merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& num_buf, gpu::gpu_mem_32u& sorted_buf, unsigned sorted_buf_size, unsigned n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::global_merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(num_buf.cuptr(), sorted_buf.cuptr(), sorted_buf_size, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
