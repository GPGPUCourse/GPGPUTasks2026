#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/primitive_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
static inline uint ExpandBits(uint v)
{
    // Ensure we have only lowest 10 bits
    rassert(v == (v & 0x3FFu), 76389413321);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
static inline MortonCode ComputeMorton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = clamp((int)(x * 1024.0f), 0, 1023);
    unsigned int iy = clamp((int)(y * 1024.0f), 0, 1023);
    unsigned int iz = clamp((int)(z * 1024.0f), 0, 1023);

    unsigned int xx = ExpandBits(ix);
    unsigned int yy = ExpandBits(iy);
    unsigned int zz = ExpandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void compute_morton_codes(
    __global const float*   min_centroid,
    __global const float*   max_centroid,
    __global const float*   centroids,
    __global PrimitiveGPU*  primitives,
    uint                    n
) {
    const uint thread_id = get_global_id(0);
    if (thread_id >= n) {
        return;
    }

    const float eps = 1e-9f;
    const float dx = fmax(max_centroid[0] - min_centroid[0], eps);
    const float dy = fmax(max_centroid[1] - min_centroid[1], eps);
    const float dz = fmax(max_centroid[2] - min_centroid[2], eps);

    const float c_x = centroids[thread_id * 3 + 0];
    const float c_y = centroids[thread_id * 3 + 1];
    const float c_z = centroids[thread_id * 3 + 2];
    float nx = (c_x - min_centroid[0]) / dx;
    float ny = (c_y - min_centroid[1]) / dy;
    float nz = (c_z - min_centroid[2]) / dz;

    // Clamp to [0,1]
    nx = fmin(fmax(nx, 0.0f), 1.0f);
    ny = fmin(fmax(ny, 0.0f), 1.0f);
    nz = fmin(fmax(nz, 0.0f), 1.0f);

    primitives[thread_id].morton_code = ComputeMorton3D(nx, ny, nz);
}