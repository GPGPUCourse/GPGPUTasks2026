#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ unsigned int warp_inclusive_scan(unsigned int value) {
    const int lane = threadIdx.x % 32;
    for (int offset = 1; offset <= 16; offset *= 2) {
        unsigned int another_value = __shfl_up_sync(-1, value, offset);
        if (lane - offset >= 0) {
            value += another_value;
        }
    }
    return value;
}

// TODO(task05): при is_input_level сканировать индикаторы нулевого бита 0
// и сразу распределять исходные числа в output. На остальных уровнях
// записывать префиксные суммы, как в task04.
// Общее число нулей доступно в последнем префиксе reduced;
// при reduced == nullptr его нужно получить из локального scan.
// Вычислительное тело дословно перенесено из task04 (63b734c).
__global__ void radix_sort_02_prefix_accumulation(
    const unsigned int* reduced, // inclusive prefixes of block sums; nullptr for the top level
    const unsigned int* input, // n elements
    unsigned int* output, // n inclusive prefixes; may alias input
    unsigned int n,
    bool is_input_level,
    unsigned int bit)
{
    __shared__ unsigned int local[BLOCK_SIZE];
    __shared__ unsigned int save[BLOCK_SIZE];
    __shared__ unsigned int local2[BLOCK_SIZE / 32];
    int total_zeros;
    if(is_input_level and reduced != nullptr){
        int reduced_length = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total_zeros = reduced[reduced_length - 1];
    }
    int base = blockIdx.x * BLOCK_SIZE;
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if (base + i < n) {
            auto val = input[base + i];
            if (is_input_level) {
                save[i] = val;
                local[i] = (val >> bit) % 2 == 0;
            }
            else {
                local[i] = val;
            }
        }
        else {
            local[i] = 0;
        }
    }
    if (threadIdx.x == 0 and reduced != nullptr and blockIdx.x > 0) {
        local[0] += reduced[blockIdx.x - 1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        unsigned int sum = local[i];
        sum = warp_inclusive_scan(sum);
        local[i] = sum;
        if (threadIdx.x % 32 == 31) {
            local2[i / 32] = sum;
        }
    }
    __syncthreads();
    {
        int i = threadIdx.x;
        if (i - i % 32 < BLOCK_SIZE / 32) {
            unsigned int sum = 0;
            if (i < BLOCK_SIZE / 32) {
                sum = local2[i];
            }
            sum = warp_inclusive_scan(sum);
            if (i < BLOCK_SIZE / 32) {
                local2[i] = sum;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if (i / 32 > 0) {
            local[i] += local2[i / 32 - 1];
        }
    }
    __syncthreads();
    if(is_input_level and reduced == nullptr){
        total_zeros = local[BLOCK_SIZE - 1];
    }
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
        if (base + i < n) {
            if (is_input_level) {
                auto val = save[i];
                int pos = 0;
                if ((val >> bit) % 2 == 0) {
                    pos = local[i] - 1;
                } else{
                    pos = total_zeros + base + i - local[i];
                }
                output[pos] = save[i];
            }
            else {
                output[base + i] = local[i];
            }
        }
    }
}

namespace cuda {
    void radix_sort_02_prefix_accumulation(const gpu::WorkSize& workSize,
        const gpu::gpu_mem_32u& reduced, const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, unsigned int n, bool is_input_level, unsigned int bit)
    {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        const unsigned int* reduced_ptr = reduced.isNull() ? nullptr : reduced.cuptr();
        ::radix_sort_02_prefix_accumulation << <workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (reduced_ptr, input.cuptr(), output.cuptr(), n, is_input_level, bit);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
