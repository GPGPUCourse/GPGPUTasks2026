#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "geometry_helpers.cu"
#include <math_constants.h>
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

__device__ inline void atomicMinFloat(float* addr, float value)
{
    int* bits = reinterpret_cast<int*>(addr);
    int old = atomicCAS(bits, 0, 0);

    while (value < __int_as_float(old)) {
        const int expected = old;
        old = atomicCAS(bits, expected, __float_as_int(value));
        if (old == expected)
            break;
    }
}

__device__ inline void atomicMaxFloat(float* addr, float value)
{
    int* bits = reinterpret_cast<int*>(addr);
    int old = atomicCAS(bits, 0, 0);

    while (value > __int_as_float(old)) {
        const int expected = old;
        old = atomicCAS(bits, expected, __float_as_int(value));
        if (old == expected)
            break;
    }
}

__global__ void find_centroid_bounds(
    const float* vertices,
    const unsigned int* faces,
    float3* bounds,
    unsigned int nfaces)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces)
        return;

    const uint3 f = loadFace(faces, i);
    const float3 v0 = loadVertex(vertices, f.x);
    const float3 v1 = loadVertex(vertices, f.y);
    const float3 v2 = loadVertex(vertices, f.z);

    const float x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    const float y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    const float z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

    atomicMinFloat(&bounds[0].x, x);
    atomicMinFloat(&bounds[0].y, y);
    atomicMinFloat(&bounds[0].z, z);

    atomicMaxFloat(&bounds[1].x, x);
    atomicMaxFloat(&bounds[1].y, y);
    atomicMaxFloat(&bounds[1].z, z);
}

__global__ void faces_morton_codes(
    const float* vertices,
    const unsigned int* faces,
    uint2* morton_codes,
    float3* bounds,
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

    const float eps = 1e-9f;
    const float dx = max(bounds[1].x - bounds[0].x, eps);
    const float dy = max(bounds[1].y - bounds[0].y, eps);
    const float dz = max(bounds[1].z - bounds[0].z, eps);

    float nx = (centroid.x - bounds[0].x) / dx;
    float ny = (centroid.y - bounds[0].y) / dy;
    float nz = (centroid.z - bounds[0].z) / dz;

    // Clamp to [0,1]
    nx = min(max(nx, 0.0f), 1.0f);
    ny = min(max(ny, 0.0f), 1.0f);
    nz = min(max(nz, 0.0f), 1.0f);

    morton_codes[index] = { morton3DGPU(nx, ny, nz), index };
}

__global__ void init_centroid_bounds(float3* bounds)
{
    bounds[0] = make_float3(CUDART_INF_F, CUDART_INF_F, CUDART_INF_F);
    bounds[1] = make_float3(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
}

namespace cuda {
void faces_morton_codes(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const gpu::shared_device_buffer_typed<uint2>& morton_codes,
    const gpu::shared_device_buffer_typed<float3>& bounds,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::init_centroid_bounds<<<1, 1, 0, stream>>>(bounds.cuptr());
    find_centroid_bounds<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), bounds.cuptr(), nfaces);

    ::faces_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), morton_codes.cuptr(),
        bounds.cuptr(),
        nfaces);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
