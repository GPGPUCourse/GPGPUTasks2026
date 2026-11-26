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

__device__ __forceinline__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ MortonCode morton3D(float x, float y, float z)
{
    unsigned int ix = min(max((int)(x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int)(y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int)(z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    return (xx << 2) | (yy << 1) | zz;
}

__global__ void build_mortons_codes(
    const float* vertices,
    const unsigned int* faces,
    MortonCode* mortons_codes,
    unsigned int nfaces,
    const int* cMin,
    const int* cMax
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

        const float eps = 1e-9f;
        const float dx = fmaxf((float)(cMax[0] - cMin[0]), eps);
        const float dy = fmaxf((float)(cMax[1] - cMin[1]), eps);
        const float dz = fmaxf((float)(cMax[2] - cMin[2]), eps);

        float nx = (c.x - (float)cMin[0]) / dx;
        float ny = (c.y - (float)cMin[1]) / dy;
        float nz = (c.z - (float)cMin[2]) / dz;

        nx = fminf(fmaxf(nx, 0.0f), 1.0f);
        ny = fminf(fmaxf(ny, 0.0f), 1.0f);
        nz = fminf(fmaxf(nz, 0.0f), 1.0f);

        mortons_codes[id] = morton3D(nx, ny, nz);
    }
}

namespace cuda {
void build_mortons_codes(
    const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u &faces,
    gpu::gpu_mem_32u &mortons_codes,
    unsigned int nfaces,
    const gpu::gpu_mem_32i &cMinGpu,
    const gpu::gpu_mem_32i &cMaxGpu)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::build_mortons_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        mortons_codes.cuptr(),
        nfaces, cMinGpu.cuptr(), cMaxGpu.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda