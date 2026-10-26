#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "radix_sort_common.cu"

__global__ void radix_sort_04_scatter(
    const unsigned int* keys,
    const unsigned int* offsets,
          unsigned int* out,
    unsigned int n,
    unsigned int bit_shift,
    unsigned int n_groups)
{
    __shared__ unsigned int rank[RADIX * SORT_WG_SIZE];
    __shared__ unsigned int scratch[SORT_WG_SIZE];
    __shared__ unsigned int digit_start[RADIX];

    const unsigned int lid = threadIdx.x;
    const unsigned int group = blockIdx.x;
    const unsigned int from = group * SORT_TILE + lid * SORT_IPT;

    unsigned int values[SORT_IPT];
    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        values[k] = (from + k < n) ? keys[from + k] : 0;
    }

    for (unsigned int b = 0; b < RADIX; ++b) {
        rank[b * SORT_WG_SIZE + lid] = 0;
    }
    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        if (from + k < n) {
            ++rank[((values[k] >> bit_shift) & RADIX_MASK) * SORT_WG_SIZE + lid];
        }
    }
    __syncthreads();

    workgroupExclusiveScan(rank, scratch, RADIX);

    for (unsigned int b = lid; b < RADIX; b += SORT_WG_SIZE) {
        digit_start[b] = rank[b * SORT_WG_SIZE];
    }
    __syncthreads();

    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        if (from + k < n) {
            const unsigned int digit = (values[k] >> bit_shift) & RADIX_MASK;
            out[offsets[digit * n_groups + group] + (rank[digit * SORT_WG_SIZE + lid]++ - digit_start[digit])] = values[k];
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &keys, const gpu::gpu_mem_32u &offsets, gpu::gpu_mem_32u &out, unsigned int n, unsigned int bit_shift, unsigned int n_groups)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(keys.cuptr(), offsets.cuptr(), out.cuptr(), n, bit_shift, n_groups);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
