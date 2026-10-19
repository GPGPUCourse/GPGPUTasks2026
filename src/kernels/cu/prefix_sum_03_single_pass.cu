#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "helpers/prefix_sum_scan.cuh"
#include "../defines.h"

namespace cuda {
unsigned int prefix_sum_03_status_size(unsigned int n)
{
    return 2u * scan::tiles_for(n) + 2u;
}

// Inclusive prefix sum of `input` into `output` in one pass over the data (see helpers/prefix_sum_scan.cuh).
// `status` is scratch: 2 * tiles + 2 unsigned ints (a 64-bit word per tile and the tile counter), zeroed here.
void prefix_sum_03_single_pass(const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, gpu::gpu_mem_32u &status, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const unsigned tiles = scan::tiles_for(n);
    rassert(input.number() >= n && output.number() >= n, 7824561230001, input.number(), output.number(), n);
    rassert(status.number() >= 2u * tiles + 2u, 7824561230002, status.number(), tiles);
    rassert(reinterpret_cast<std::uintptr_t>(status.cuptr()) % 8 == 0, 7824561230003);
    unsigned long long* words = reinterpret_cast<unsigned long long*>(status.cuptr());
    unsigned* counter = status.cuptr() + 2u * tiles;
    CUDA_SAFE_CALL(cudaMemsetAsync(status.cuptr(), 0, sizeof(unsigned) * (2u * tiles + 2u), stream));
    scan::scan_single_pass<scan::CFG_STRIPED, scan::CFG_WARP_LOOKBACK, scan::CFG_MIN_BLOCKS><<<tiles, scan::THREADS, 0, stream>>>(
        input.cuptr(), output.cuptr(), n, words, counter);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
