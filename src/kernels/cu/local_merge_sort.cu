#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

#include "pos_bin_search.cuh"

__global__ void local_merge_sort(
    const unsigned* num_buf,
    unsigned* sorted_buf,
    unsigned n)
{
    unsigned loc_id = threadIdx.x;
    unsigned wg_id = blockIdx.x;
    unsigned glob_id = wg_id * GROUP_SIZE + loc_id;

    __shared__ unsigned loc_buf_1[GROUP_SIZE];
    __shared__ unsigned loc_buf_2[GROUP_SIZE];
    loc_buf_1[loc_id] = (glob_id < n) ? num_buf[glob_id] : ~0U;
    __syncthreads();

    unsigned* loc_num_buf = loc_buf_1;
    unsigned* loc_sorted_buf = loc_buf_2;

    for (unsigned sorted_buf_size = 2; sorted_buf_size <= GROUP_SIZE; sorted_buf_size <<= 1) {
        unsigned chunk_size = sorted_buf_size / 2;
        unsigned chunk_id = loc_id / chunk_size;
        unsigned chunk_start = chunk_id * chunk_size;

        chunk_side cs = (chunk_id % 2 == 0) ? chunk_side::left : chunk_side::right;

        unsigned neigh_chunk_start = (cs == chunk_side::right) ? (chunk_start - chunk_size) : (chunk_start + chunk_size);

        unsigned pos_in_neigh_chunk = pos_bin_search(
            &(loc_num_buf[neigh_chunk_start]), loc_num_buf[loc_id], chunk_size, cs);
        unsigned pos_in_chunk = loc_id - chunk_start;
        unsigned pos_in_loc_sorted_buf = pos_in_chunk + pos_in_neigh_chunk;

        unsigned loc_sorted_buf_offset = chunk_start;
        if (cs == chunk_side::right)
            loc_sorted_buf_offset -= chunk_size;
        pos_in_loc_sorted_buf += loc_sorted_buf_offset;

        loc_sorted_buf[pos_in_loc_sorted_buf] = loc_num_buf[loc_id];

        unsigned* tmp = loc_num_buf;
        loc_num_buf = loc_sorted_buf;
        loc_sorted_buf = tmp;

        __syncthreads();
    }

    if (glob_id < n)
        sorted_buf[glob_id] = loc_num_buf[loc_id];
}

namespace cuda {
void local_merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& num_buf, gpu::gpu_mem_32u& sorted_buf, unsigned n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::local_merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(num_buf.cuptr(), sorted_buf.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
