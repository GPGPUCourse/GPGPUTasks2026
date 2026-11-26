#include "../kernels.h"

#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"
#include "../shared_structs/morton_code_gpu_shared.h"

#include <climits>

__global__ void init_bounds_int(int* cMin, int* cMax)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int k = 0; k < 3; ++k) {
            cMin[k] = INT_MAX;
            cMax[k] = INT_MIN;
        }
    }
}

__global__ void calculate_bounding_box(
    const float* vertices,
    const unsigned int* faces,
    unsigned int nfaces,
    int* cMin,
    int* cMax
)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < nfaces) {
        uint3 face = loadFace(faces, id);
        float3 v0 = loadVertex(vertices, face.x);
        float3 v1 = loadVertex(vertices, face.y);
        float3 v2 = loadVertex(vertices, face.z);

        float3 c;
        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        atomicMin(&cMin[0], __float2int_rd(c.x));
        atomicMin(&cMin[1], __float2int_rd(c.y));
        atomicMin(&cMin[2], __float2int_rd(c.z));

        atomicMax(&cMax[0], __float2int_ru(c.x));
        atomicMax(&cMax[1], __float2int_ru(c.y));
        atomicMax(&cMax[2], __float2int_ru(c.z));
    }
}

namespace cuda {
void calculate_bounding_box(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u &faces,
    unsigned int nfaces,
    gpu::gpu_mem_32i &cMinGpu,
    gpu::gpu_mem_32i &cMaxGpu)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::init_bounds_int<<<1, 1, 0, stream>>>(cMinGpu.cuptr(), cMaxGpu.cuptr());
    CUDA_CHECK_KERNEL(stream);

    ::calculate_bounding_box<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        nfaces, cMinGpu.cuptr(), cMaxGpu.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda