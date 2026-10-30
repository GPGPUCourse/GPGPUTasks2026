#include <cstddef>

#include <libgpu/context.h>
#include <libgpu/cuda/cu/common.cu>
#include <libgpu/shared_device_buffer.h>

#include "helpers/rassert.cu"
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void merge_sort_local(
    const unsigned int* input_data,
    unsigned int* output_data,
    unsigned int input_data_size)
{
    const unsigned int local_array_size = 256;

    unsigned int task_index = blockIdx.x;
    unsigned int input_offset = task_index * local_array_size;
    if (input_offset >= input_data_size) {
        return;
    }

    unsigned int i = threadIdx.x;
    unsigned int local_size = min(local_array_size, input_data_size - input_offset);

    __shared__ unsigned int first[local_array_size];
    __shared__ unsigned int second[local_array_size];

    if (i < local_size) {
        first[i] = input_data[input_offset + i];
    }
    __syncthreads();

    unsigned int* input_local = first;
    unsigned int* output_local = second;

    for (unsigned int merge_array_size = 2; merge_array_size <= local_array_size; merge_array_size *= 2) {
        unsigned int first_start = i * merge_array_size;

        if (first_start < local_size) {
            unsigned int second_start = min(first_start + merge_array_size / 2, local_size);
            unsigned int end = min(first_start + merge_array_size, local_size);

            unsigned int l = first_start;
            unsigned int r = second_start;

            for (unsigned int k = first_start; k < end; ++k) {
                if (l < second_start && (r == end || input_local[l] <= input_local[r])) {
                    output_local[k] = input_local[l++];
                } else {
                    output_local[k] = input_local[r++];
                }
            }
        }

        __syncthreads();

        unsigned int* temp = input_local;
        input_local = output_local;
        output_local = temp;
    }

    if (i < local_size) {
        output_data[input_offset + i] = input_local[i];
    }
}
namespace cuda {
void merge_sort_local(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, unsigned int input_data_size)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_local<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), input_data_size);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
