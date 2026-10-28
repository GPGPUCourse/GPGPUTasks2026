#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

constexpr int TILE = 256;

__device__ int find_partition(
    const unsigned int* A, int na,
    const unsigned int* B, int nb,
    int d)
{
    int l = max(0, d - nb);
    int r = min(d, na);

    while (l < r) {
        int x = l + (r - l) / 2;
        int y = d - x;

        if (y > 0 && x < na && B[y - 1] >= A[x]) {
            l = x + 1;
        } else {
            r = x;
        }
    }

    return l;
}

__global__ void merge_sort_points(
    const unsigned int* input,
    unsigned int* points,
    int sorted_k,
    int n)
{
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_count = (n + TILE - 1) / TILE;

    if (tile >= tile_count)
        return;

    int sorted = 1 << sorted_k;
    int pair_size = sorted << 1;

    int d_global = tile * TILE;
    int base = (d_global / pair_size) * pair_size;

    int na = min(sorted, n - base);
    int nb = min(sorted, n - base - na);

    const unsigned int* A = input + base;
    const unsigned int* B = A + na;

    int d = d_global - base;

    points[tile] = find_partition(A, na, B, nb, d);
}

namespace cuda {
void merge_sort_points(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_points<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
