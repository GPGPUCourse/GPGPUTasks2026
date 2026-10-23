#include "../kernels.h"

// Parse the course wrappers before CCCL introduces cuda::std (the wrappers
// use unqualified std:: names inside their own namespace cuda).
#include <libgpu/cuda/utils.h>
#include <cub/device/device_radix_sort.cuh>

namespace cuda {
namespace {
void check_buffers(const gpu::gpu_mem_32u& input, const gpu::gpu_mem_32u& output, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 82050001);
    rassert(n >= 0, 82050002, n);
    rassert(input.number() >= static_cast<size_t>(n), 82050003);
    rassert(output.number() >= static_cast<size_t>(n), 82050004);
}
} // namespace

size_t cub_radix_sort_temp_bytes(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, int n)
{
    check_buffers(input, output, n);
    if (n == 0) {
        return 0;
    }

    size_t bytes = 0;
    const uint32_t* keys_in = input.cuptr();
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortKeys(
        nullptr, bytes, keys_in, output.cuptr(), n, 0, 32, gpu::Context().cudaStream()));
    return bytes;
}

void cub_radix_sort(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, gpu::gpu_mem_8u& scratch, int n)
{
    check_buffers(input, output, n);
    if (n == 0) {
        return;
    }

    // Use CUB's non-overwriting pointer API, not DoubleBuffer: every invocation
    // sorts the original input and writes the complete result to output.
    // No allocation, host transfer, result caching or input-dependent bit range.
    rassert(!scratch.isNull(), 82050005);
    size_t bytes = scratch.size();
    const uint32_t* keys_in = input.cuptr();
    cudaStream_t stream = gpu::Context().cudaStream();
    CUDA_SAFE_CALL(cub::DeviceRadixSort::SortKeys(
        scratch.cuptr(), bytes, keys_in, output.cuptr(), n, 0, 32, stream));
    // Include ALL device work in the caller's wall-clock timing and propagate
    // asynchronous errors, without synchronizing between CUB's internal passes.
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
}
} // namespace cuda
