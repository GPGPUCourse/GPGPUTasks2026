#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"
#include "../helper.h"
// FROM GPUGEMS

#define NUM_BANKS 32
#define uint4_0 make_uint4(0, 0, 0, 0)

__device__ __forceinline__ unsigned int compute_index(unsigned int n)
{
    return n + n / NUM_BANKS;
}


__global__ void prefix_sum_01_reduction_blelloch(
    uint4* data, // contains n values
    uint4* res,
    uint4* partial_sums,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ uint4 temp[PREFIX_SUM_GROPU_DATA_SIZE + (PREFIX_SUM_GROPU_DATA_SIZE - 1) / NUM_BANKS];

    if (index < n) {
        temp[compute_index(threadIdx.x)] = data[index];
    } else {
        temp[compute_index(threadIdx.x)] = uint4_0;
    }
    const unsigned int extra_index = index + GROUP_SIZE;
    if (extra_index < n) {
        temp[compute_index(threadIdx.x + GROUP_SIZE)] = data[extra_index];
    } else {
        temp[compute_index(threadIdx.x + GROUP_SIZE)] = uint4_0;
    }

    unsigned int offset = 1;

#pragma unroll
    for (unsigned int d = PREFIX_SUM_GROPU_DATA_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadIdx.x < d) {
            unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
            unsigned int bi = ai + offset;
            ai = compute_index(ai);
            bi = compute_index(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();

    const uint4 sum = temp[compute_index(PREFIX_SUM_GROPU_DATA_SIZE - 1)];
    __syncthreads();
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
        temp[compute_index(PREFIX_SUM_GROPU_DATA_SIZE - 1)] = uint4_0;
    }

#pragma unroll
    for (unsigned int d = 1; d < PREFIX_SUM_GROPU_DATA_SIZE; d <<= 1) {
        __syncthreads();
        offset >>= 1;

        if (threadIdx.x < d) {

            unsigned int first = (2 * threadIdx.x + 1) * offset - 1;
            unsigned int second = first + offset;
            first = compute_index(first);
            second = compute_index(second);

            uint4 first_buf = temp[first];
            temp[first] = temp[second];
            temp[second] += first_buf;
        }
    }

    __syncthreads();

    if (index < n) {
        res[index] = temp[compute_index(threadIdx.x + 1)];
    }

    if (index + GROUP_SIZE < n) {
        res[index + GROUP_SIZE] = (threadIdx.x == GROUP_SIZE - 1) ? sum : temp[compute_index(threadIdx.x + 1 + GROUP_SIZE)];
    }
}

namespace cuda {
void prefix_sum_01_reduction_blelloch(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<uint4>& data, gpu::shared_device_buffer_typed<uint4>& res, gpu::shared_device_buffer_typed<uint4>& partial_sums, unsigned int data_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction_blelloch<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data.cuptr(), res.cuptr(), partial_sums.cuptr(), data_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda