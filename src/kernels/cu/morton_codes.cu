#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include <cuda_runtime.h>
#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

__device__ unsigned int morton_code(float x, float y, float z)
{
    constexpr unsigned int cells = 1u << MORTON_BITS_PER_AXIS;
    constexpr float maxCoord = float(cells - 1);

    unsigned int ix = (unsigned int)fminf(fmaxf(x * cells, 0.0f), maxCoord);
    unsigned int iy = (unsigned int)fminf(fmaxf(y * cells, 0.0f), maxCoord);
    unsigned int iz = (unsigned int)fminf(fmaxf(z * cells, 0.0f), maxCoord);

    unsigned int code = 0;

#pragma unroll
    for (int bit = 0; bit < MORTON_BITS_PER_AXIS; ++bit) {
        code |= ((ix >> bit) & 1u) << (3 * bit + 2);
        code |= ((iy >> bit) & 1u) << (3 * bit + 1);
        code |= ((iz >> bit) & 1u) << (3 * bit);
    }

    return code;
}

__global__ void generate_morton_codes(
    const float* centroids,
    const AABBGPU* bounds,
    unsigned int* codes,
    unsigned int* triangleIndices,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const AABBGPU& box = bounds[0];

    float dx = fmaxf(box.max_x - box.min_x, 1e-9f);
    float dy = fmaxf(box.max_y - box.min_y, 1e-9f);
    float dz = fmaxf(box.max_z - box.min_z, 1e-9f);

    float x = (centroids[3 * i] - box.min_x) / dx;
    float y = (centroids[3 * i + 1] - box.min_y) / dy;
    float z = (centroids[3 * i + 2] - box.min_z) / dz;

    codes[i] = morton_code(x, y, z);
    triangleIndices[i] = i;
}

namespace cuda {

void prepare_morton_codes(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& centroids,
    const gpu::shared_device_buffer_typed<AABBGPU>& bounds,
    gpu::gpu_mem_32u& codes,
    gpu::gpu_mem_32u& triangleIndices,
    int n)
{
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();

    ::generate_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        centroids.cuptr(),
        bounds.cuptr(),
        codes.cuptr(),
        triangleIndices.cuptr(),
        n);

    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        codes.cuptr(),
        codes.cuptr() + n,
        triangleIndices.cuptr());

    CUDA_CHECK_KERNEL_SYNC(stream);
}

}