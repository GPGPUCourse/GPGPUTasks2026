#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../blelloch_local_sort.h"
#include "../defines.h"
#include "../helper.h"
// FROM GPUGEMS


__global__ void prefix_sum_01_reduction_blelloch(
    unsigned int* data, // contains n values
    unsigned int* res,
    unsigned int* partial_sums,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    __shared__ unsigned int temp[PREFIX_SUM_GROPU_DATA_SIZE + (PREFIX_SUM_GROPU_DATA_SIZE - 1) / NUM_BANKS];

    if (index < n) {
        temp[compute_index(threadIdx.x)] = data[index];
    } else {
        temp[compute_index(threadIdx.x)] = 0;
    }
    const unsigned int extra_index = index + SCAN_GROUP_SIZE;
    if (extra_index < n) {
        temp[compute_index(threadIdx.x + SCAN_GROUP_SIZE)] = data[extra_index];
    } else {
        temp[compute_index(threadIdx.x + SCAN_GROUP_SIZE)] = 0;
    }

    blelloch_local_scan(temp, partial_sums, PREFIX_SUM_GROPU_DATA_SIZE);

    if (index < n) {
        res[index] = temp[compute_index(threadIdx.x)];
    }

    if (index + SCAN_GROUP_SIZE < n) {
        res[index + SCAN_GROUP_SIZE] = temp[compute_index(threadIdx.x + SCAN_GROUP_SIZE)];
    }
}

namespace cuda {
void prefix_sum_01_reduction_blelloch(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& data, const gpu::gpu_mem_32u& res, const gpu::gpu_mem_32u& partial_sums, unsigned int data_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction_blelloch<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(data.cuptr(), res.cuptr(), partial_sums.cuptr(), data_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda