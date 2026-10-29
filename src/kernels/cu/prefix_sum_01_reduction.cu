#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* pow2_sum, 
          unsigned int* next_pow2_sum,
    unsigned int sz,
    unsigned int sft
)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (sz + 1) / 2) return;

    int res = pow2_sum[2 * idx + sft] + pow2_sum[2 * idx + 1 + sft];
    next_pow2_sum[idx + sft + sz] = res;
    // if (idx < 10 && sft == 0) {
    //     printf("idx %d val %d\n", idx + sft + sz, res);
    // }
}

// __global__ void prefix_sum_01_reduction(
//     // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
//     // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
//     const unsigned int* pow2_sum, // contains n values
//           unsigned int* next_pow2_sum, // will contain (n+1)/2 values
//     unsigned int n)
// {
//     uint idx = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ int lines[GROUP_SIZE];
//     lines[threadIdx.x] = idx < n ? pow2_sum[idx] : 0;
//     __syncthreads();
//     int res = 0;
//     if (threadIdx.x % 2 < 32) {
//         for (int i = 0; i < 8; ++i){
//             res += lines[i + threadIdx.x * 8];
//             lines[i + threadIdx.x * 8] = res;
//         }
//     }
//     __syncthreads();

//     accum[idx] = res;
//     if (idx % 2 == 1) {
//         buffer[idx / 2] = lines[threadIdx.x] + lines[threadIdx.x - 1];
//     }
// }

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n, unsigned int sft)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n, sft);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
