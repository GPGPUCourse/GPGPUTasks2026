#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values, const unsigned int* prefixes, unsigned int* output,
    size_t n, unsigned int shift)
{
    __shared__ unsigned int warp_counts[RADIX_BUCKETS][GROUP_SIZE / 32];
    __shared__ unsigned int bucket_starts[RADIX_BUCKETS];
    __shared__ unsigned int ordered[GROUP_SIZE];

    unsigned int thread = threadIdx.x;
    unsigned int lane = thread % 32;
    unsigned int warp = thread / 32;
    unsigned int i = blockIdx.x * blockDim.x + thread;
    unsigned int value = i < n ? values[i] : 0;
    unsigned int digit = (value >> shift) & (RADIX_BUCKETS - 1);
    unsigned int valid = __ballot_sync(0xffffffff, i < n);
    unsigned int peers = __match_any_sync(0xffffffff, digit) & valid;

    if (thread < RADIX_BUCKETS * (GROUP_SIZE / 32)) {
        warp_counts[thread / (GROUP_SIZE / 32)][thread % (GROUP_SIZE / 32)] = 0;
    }
    __syncthreads();
    if (i < n && lane == __ffs(peers) - 1) {
        warp_counts[digit][warp] = __popc(peers);
    }
    __syncthreads();

    if (warp == 0) {
        unsigned int count = 0;
        if (lane < RADIX_BUCKETS) {
            for (unsigned int w = 0; w < GROUP_SIZE / 32; ++w) {
                count += warp_counts[lane][w];
            }
        }
        unsigned int sum = count;
        for (unsigned int offset = 1; offset < 32; offset *= 2) {
            unsigned int left = __shfl_up_sync(0xffffffff, sum, offset);
            if (lane >= offset) {
                sum += left;
            }
        }
        if (lane < RADIX_BUCKETS) {
            bucket_starts[lane] = sum - count;
        }
    }
    __syncthreads();

    if (i < n) {
        unsigned int rank = __popc(peers & ((1u << lane) - 1));
        for (unsigned int w = 0; w < warp; ++w) {
            rank += warp_counts[digit][w];
        }
        ordered[bucket_starts[digit] + rank] = value;
    }
    __syncthreads();

    if (i < n) {
        value = ordered[thread];
        digit = (value >> shift) & (RADIX_BUCKETS - 1);
        unsigned int bucket = digit * gridDim.x + blockIdx.x;
        unsigned int start = bucket == 0 ? 0 : prefixes[bucket - 1];
        output[start + thread - bucket_starts[digit]] = value;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &prefixes,
    gpu::gpu_mem_32u &output, unsigned int shift)
{
    if (values.number() == 0) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize workSize(GROUP_SIZE, values.number());
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), prefixes.cuptr(), output.cuptr(), values.number(), shift);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
