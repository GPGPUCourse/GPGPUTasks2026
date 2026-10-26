#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "radix_sort_common.cu"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
          unsigned int* data,
    const unsigned int* block_offsets,
    unsigned int n,
    unsigned int use_offsets)
{
    __shared__ unsigned int tile[SCAN_TILE];
    __shared__ unsigned int scratch[SCAN_WG_SIZE];

    const unsigned int lid = threadIdx.x;
    const unsigned int base = blockIdx.x * SCAN_TILE;

    for (unsigned int k = 0; k < SCAN_IPT; ++k) {
        const unsigned int p = lid + k * SCAN_WG_SIZE;
        tile[p] = (base + p < n) ? data[base + p] : 0;
    }
    __syncthreads();

    workgroupExclusiveScan(tile, scratch, SCAN_IPT);

    const unsigned int add = use_offsets ? block_offsets[blockIdx.x] : 0;
    for (unsigned int k = 0; k < SCAN_IPT; ++k) {
        const unsigned int p = lid + k * SCAN_WG_SIZE;
        if (base + p < n) {
            data[base + p] = tile[p] + add;
        }
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &data, const gpu::gpu_mem_32u &block_offsets, unsigned int n, unsigned int use_offsets)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data.cuptr(), block_offsets.cuptr(), n, use_offsets);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
