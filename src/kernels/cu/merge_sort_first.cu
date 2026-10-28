#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

constexpr int TILE = 256;
constexpr int TILE_LOG = 8;

__global__ void merge_sort_first(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    __shared__ unsigned int a[TILE];
    __shared__ unsigned int b[TILE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * TILE + tid;


    a[tid] = (i < n) ? input_data[i] : UINT_MAX;

    __syncthreads();

    unsigned int* in = a;
    unsigned int* out = b;

    for (int k = 0; k < TILE_LOG; ++k) {
        const int sorted = 1 << k;
        const int workSize = sorted << 1;

        const int real_i = tid % workSize;
        const int offset = tid - real_i;
        const unsigned int value = in[tid];

        int pos;

        if (real_i >= sorted) {

            int l = offset;
            int r = offset + sorted;

            while (l < r) {
                int mid = (l + r) >> 1;

                if (in[mid] <= value) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }

            pos = tid - sorted + l - offset;
        } else {

            const int start = offset + sorted;
            int l = start;
            int r = offset + workSize;

            while (l < r) {
                int mid = (l + r) >> 1;

                if (in[mid] < value) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }

            pos = tid + l - start;
        }

        out[pos] = value;

        __syncthreads();

        unsigned int* tmp = in;
        in = out;
        out = tmp;
    }

    if (i < n) {
        output_data[i] = in[tid];
    }
}

namespace cuda {
void merge_sort_first(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort_first<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda\