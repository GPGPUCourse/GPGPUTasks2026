#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include "helpers/merge_sort_impl.cuh"

namespace cuda {
// sorted_k == 0: block tile sort (sorted_k template parameter is unused below TILE)
// sorted_k >= MERGE_TILE: one pairwise merge pass, runs of sorted_k become runs of 2*sorted_k
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    if (sorted_k == 0) {
        merge6::sort_tiles_async<GROUP_SIZE, MERGE_ITEMS_PER_THREAD>(input_data.cuptr(), output_data.cuptr(), n, stream);
    } else {
        merge6::merge_pass_async<GROUP_SIZE, MERGE_ITEMS_PER_THREAD>(input_data.cuptr(), output_data.cuptr(), sorted_k, n, stream);
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
