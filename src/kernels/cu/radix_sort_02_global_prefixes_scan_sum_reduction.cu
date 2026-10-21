#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

#include "../helper.h"

__device__ __forceinline__ void count_digit(
    unsigned int key,
    unsigned int offset,
    unsigned char& a0,
    unsigned char& a1,
    unsigned char& a2,
    unsigned char& a3)
{
    const unsigned int b = (key >> offset) & 3u;
    a0 += (b == 0);
    a1 += (b == 1);
    a2 += (b == 2);
    a3 += (b == 3);
}

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    uint4* counters,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    unsigned char a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    if (index + 3 < n) {
        uint4 data = reinterpret_cast<const uint4*>(input)[index / 4];
        count_digit(data.x, offset, a0, a1, a2, a3);
        count_digit(data.y, offset, a0, a1, a2, a3);
        count_digit(data.z, offset, a0, a1, a2, a3);
        count_digit(data.w, offset, a0, a1, a2, a3);
    } else {
        if (index < n) {
            a0 += ((input[index] >> offset) & 0x3) == 0;
        }
        if (index + 1 < n) {
            a1 += ((input[index + 1] >> offset) & 0x3) == 0;
        }
        if (index + 2 < n) {
            a2 += ((input[index + 2] >> offset) & 0x3) == 0;
        }
    }

    __shared__ unsigned int counter[4];
    if (threadIdx.x < 4) {
        counter[threadIdx.x] = 0;
    }

    __syncthreads();

    atomicAdd(counter, a0);
    atomicAdd(counter + 1, a1);
    atomicAdd(counter + 2, a2);
    atomicAdd(counter + 3, a3);

    __syncthreads();
    uint4 res = make_uint4(counter[0], counter[1], counter[2], counter[3]);
    if (threadIdx.x == 0) {
        counters[blockIdx.x] = res;
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::shared_device_buffer_typed<uint4>& counters, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), counters.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
