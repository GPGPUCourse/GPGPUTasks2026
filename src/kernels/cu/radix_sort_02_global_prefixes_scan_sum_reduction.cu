#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(const uint32_t *input, uint32_t *output)
{
    const size_t baseIdx = blockIdx.x * GROUP_SIZE;

    uint4 a = *reinterpret_cast<const uint4 *>(input + baseIdx + threadIdx.x * 4);
    uint4 b = *reinterpret_cast<const uint4 *>(input + baseIdx + GROUP_SIZE / 2 + threadIdx.x * 4);

    uint32_t sumA = a.x + a.y + a.z + a.w;
    uint32_t sumB = b.x + b.y + b.z + b.w;

    uint32_t prefixA = sumA;
    uint32_t prefixB = sumB;

#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
        uint32_t prevA = __shfl_up_sync(0xFFFFFFFF, prefixA, offset);
        uint32_t prevB = __shfl_up_sync(0xFFFFFFFF, prefixB, offset);

        if (threadIdx.x >= offset)
        {
            prefixA += prevA;
            prefixB += prevB;
        }
    }

    uint32_t totalA = __shfl_sync(0xFFFFFFFF, prefixA, 31);

    uint32_t offsetA = prefixA - sumA;
    uint32_t offsetB = totalA + prefixB - sumB;

    uint4 resultA = make_uint4(offsetA, offsetA + a.x, offsetA + a.x + a.y, offsetA + a.x + a.y + a.z);
    uint4 resultB = make_uint4(offsetB, offsetB + b.x, offsetB + b.x + b.y, offsetB + b.x + b.y + b.z);

    *reinterpret_cast<uint4 *>(output + baseIdx + threadIdx.x * 4) = resultA;
    *reinterpret_cast<uint4 *>(output + baseIdx + GROUP_SIZE / 2 + threadIdx.x * 4) = resultB;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize &workSize, const uint32_t *input, uint32_t *output)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input, output);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
