#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

constexpr int TILE = 256;
constexpr int TILE_LOG = 8;

__global__ void merge_sort_small(
    const unsigned int* input,
    unsigned int* output,
    const unsigned int* points,
    int sorted_k,
    int n)
{
    __shared__ unsigned int A[TILE];
    __shared__ unsigned int B[TILE];

    int tile = blockIdx.x;
    int tid = threadIdx.x;

    int begin = tile * TILE;
    if (begin >= n)
        return;

    int end = min(begin + TILE, n);

    int sorted = 1 << sorted_k;
    int pair_size = 2 * sorted;
    int base = (begin / pair_size) * pair_size;

    int na = min(sorted, n - base);
    int nb = min(sorted, n - base - na);

    int d0 = begin - base;
    int d1 = end - base;

    int x0 = points[tile];
    int y0 = d0 - x0;

    int x1 = (d1 == na + nb) ? na: points[tile + 1];

    int y1 = d1 - x1;

    int countA = x1 - x0;
    int countB = y1 - y0;
    int count = countA + countB;

    if (tid < countA)
        A[tid] = input[base + x0 + tid];

    if (tid < countB)
        B[tid] = input[base + na + y0 + tid];

    __syncthreads();

    if (tid >= count)
        return;

    unsigned int value;
    int pos;

    if (tid < countA) {
        value = A[tid];

        int l = 0, r = countB;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (B[mid] < value)
                l = mid + 1;
            else
                r = mid;
        }

        pos = tid + l;
    } else {
        int j = tid - countA;
        value = B[j];

        int l = 0, r = countA;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (A[mid] <= value)
                l = mid + 1;
            else
                r = mid;
        }

        pos = j + l;
    }

    output[begin + pos] = value;
}

namespace cuda {
void merge_sort_small(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, const gpu::gpu_mem_32u& points, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_small<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), points.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
