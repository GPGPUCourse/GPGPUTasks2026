#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

#include "../helper.h"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* input,
    unsigned int* counters,
    unsigned int offset,
    unsigned int n)
{

    auto uint4_data = reinterpret_cast<const uint4*>(input);
    __shared__ unsigned int local_counters[BUCKET_COUNT];
    for (unsigned int b = threadIdx.x; b < BUCKET_COUNT; b += COUNTER_REDUCTION_GROUP_SIZE) {
        local_counters[b] = 0;
    }
    __syncthreads();
    unsigned int group_offset = blockIdx.x * blockDim.x * UINT4_PER_THREAD_IN_REDUCTION * 4u;

    for (unsigned char uint4_index = 0; uint4_index < UINT4_PER_THREAD_IN_REDUCTION; uint4_index++) {
        unsigned int read_index = group_offset + uint4_index * 4 * COUNTER_REDUCTION_GROUP_SIZE + threadIdx.x * 4;
        if (read_index + 3 < n) {
            uint4 data = uint4_data[read_index / 4];
            atomicAdd(local_counters + get_bucket(data.x, offset), 1);
            atomicAdd(local_counters + get_bucket(data.y, offset), 1);
            atomicAdd(local_counters + get_bucket(data.z, offset), 1);
            atomicAdd(local_counters + get_bucket(data.w, offset), 1);
        } else {
            while (read_index < n) {
                unsigned int data = input[read_index];
                atomicAdd(local_counters + get_bucket(data, offset), 1);
                read_index++;
            }
        }
    }

    __syncthreads();
    auto reduced_groups_count = get_reducted_groups_count(n);
    // threadIdx.x === get_bucket
    // TODO transpose???
    for (unsigned int b = threadIdx.x; b < BUCKET_COUNT; b += COUNTER_REDUCTION_GROUP_SIZE) {
        counters[b * reduced_groups_count + blockIdx.x] = local_counters[b];
    }
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& counters, unsigned int offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), counters.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
