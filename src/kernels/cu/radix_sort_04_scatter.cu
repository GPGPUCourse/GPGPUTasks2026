#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "radix_sort_common.cu"

static __device__ void localStableSortStep(unsigned int *tile, unsigned int *rank, unsigned int *scratch, unsigned int bit_shift)
{
    const unsigned int lid = threadIdx.x;

    unsigned int values[SORT_IPT];
    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        values[k] = tile[lid * SORT_IPT + k];
    }

    for (unsigned int b = 0; b < SUB_RADIX; ++b) {
        rank[b * SORT_WG_SIZE + lid] = 0;
    }
    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        ++rank[((values[k] >> bit_shift) & SUB_MASK) * SORT_WG_SIZE + lid];
    }
    __syncthreads();

    workgroupExclusiveScan(rank, scratch, SUB_RADIX);

    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        const unsigned int b = (values[k] >> bit_shift) & SUB_MASK;
        tile[rank[b * SORT_WG_SIZE + lid]++] = values[k];
    }
    __syncthreads();
}

__global__ void radix_sort_04_scatter(
    const unsigned int* keys,
    const unsigned int* offsets,
          unsigned int* out,
    unsigned int n,
    unsigned int bit_shift,
    unsigned int n_groups)
{
    __shared__ unsigned int tile[SORT_TILE];
    __shared__ unsigned int rank[SUB_RADIX * SORT_WG_SIZE];
    __shared__ unsigned int scratch[SORT_WG_SIZE];
    __shared__ unsigned int digit_start[RADIX];

    const unsigned int lid = threadIdx.x;
    const unsigned int group = blockIdx.x;
    const unsigned int base = group * SORT_TILE;
    const unsigned int tile_n = (n - base < SORT_TILE) ? (n - base) : SORT_TILE;

    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        const unsigned int p = lid + k * SORT_WG_SIZE;
        tile[p] = (p < tile_n) ? keys[base + p] : 0xFFFFFFFFu;
    }
    __syncthreads();

    for (unsigned int s = 0; s < SUB_NPASSES; ++s) {
        localStableSortStep(tile, rank, scratch, bit_shift + s * SUB_BITS);
    }

    digit_start[lid] = 0;
    __syncthreads();

    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        const unsigned int p = lid + k * SORT_WG_SIZE;
        const unsigned int digit = (tile[p] >> bit_shift) & RADIX_MASK;
        if (p == 0 || ((tile[p - 1] >> bit_shift) & RADIX_MASK) != digit) {
            digit_start[digit] = p;
        }
    }
    __syncthreads();

    for (unsigned int k = 0; k < SORT_IPT; ++k) {
        const unsigned int p = lid + k * SORT_WG_SIZE;
        if (p < tile_n) {
            const unsigned int value = tile[p];
            const unsigned int digit = (value >> bit_shift) & RADIX_MASK;
            out[offsets[digit * n_groups + group] + (p - digit_start[digit])] = value;
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
