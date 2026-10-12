#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ __forceinline__
void add4(unsigned int* output, size_t idx, size_t n, unsigned int offset)
{
    if (idx < n && n - idx >= 4)
    {
        uint4 v = *reinterpret_cast<const uint4*>(output + idx);

        v.x += offset;
        v.y += offset;
        v.z += offset;
        v.w += offset;

        *reinterpret_cast<uint4*>(output + idx) = v;
        return;
    }

    if (idx < n)     output[idx] += offset;
    if (idx + 1 < n) output[idx + 1] += offset;
    if (idx + 2 < n) output[idx + 2] += offset;
    if (idx + 3 < n) output[idx + 3] += offset;
}

__global__ void uniform_add(unsigned int* res, const unsigned int* incr, unsigned int n)
{
    constexpr unsigned int B = GROUP_SIZE * 8;

    unsigned int block = blockIdx.x + 1;

    unsigned int thid = threadIdx.x;

    unsigned int offset = incr[block - 1];

    size_t base = static_cast<size_t>(block) * B + thid * 4;

    add4(res, base, n, offset);
    add4(res, base + 4 * GROUP_SIZE, n, offset);

// #pragma unroll
//     for (unsigned int i = 0; i < 8; ++i)
//     {
//         size_t idx = base + i * GROUP_SIZE;
//
//         if (idx < n)
//             res[idx] += offset;
//     }
}


// !!! WARNING !!!
// This file is not actually implements prefix accumulation
// In fact, it performs uniform add incr[i] to every element of res' i - 1 block
namespace cuda {
void uniform_add(const gpu::WorkSize &workSize, unsigned int *res, const unsigned int *incr, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::uniform_add<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(res, incr, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
