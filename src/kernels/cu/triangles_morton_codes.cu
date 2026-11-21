#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "geometry_helpers.cu"

__device__ unsigned int expandBitsGPU(unsigned int v)
{

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
__device__ unsigned int morton3DGPU(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = cuda::min(cuda::max((int)(x * 1024.0f), 0), 1023);
    unsigned int iy = cuda::min(cuda::max((int)(y * 1024.0f), 0), 1023);
    unsigned int iz = cuda::min(cuda::max((int)(z * 1024.0f), 0), 1023);

    unsigned int xx = expandBitsGPU(ix);
    unsigned int yy = expandBitsGPU(iy);
    unsigned int zz = expandBitsGPU(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__global__ void faces_morton_codes(
    const float* vertices,
    const unsigned int* faces,
    uint2* morton_codes,
    unsigned int nfaces)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= nfaces) {
        return;
    }

    uint3 f = loadFace(faces, index);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 centroid;
    centroid.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    centroid.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    centroid.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

    morton_codes[index] = { morton3DGPU(centroid.x, centroid.y, centroid.z), index };
}

namespace cuda {
void faces_morton_codes(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const gpu::shared_device_buffer_typed<uint2>& morton_codes,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::faces_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), morton_codes.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
