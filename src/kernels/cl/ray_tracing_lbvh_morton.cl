#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
inline unsigned int expandBits(unsigned int v)
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
inline MortonCode morton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void ray_tracing_lbvh_morton(
    __global const float3* centroids,
    uint n,
    __global const AABBGPU* centroids_aabb,
    __global MortonCode* morton
) {
    const uint gid = get_global_id(0);

    if (gid >= n) return;
    
    AABBGPU aabb = *centroids_aabb;

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    const float eps = 1e-9f;
    const float dx = fmax(aabb.max_x - aabb.min_x, eps);
    const float dy = fmax(aabb.max_y - aabb.min_y, eps);
    const float dz = fmax(aabb.max_z - aabb.min_z, eps);

    const float3 c = centroids[gid];
    float nx = (c.x - aabb.min_x) / dx;
    float ny = (c.y - aabb.min_y) / dy;
    float nz = (c.z - aabb.min_z) / dz;

    // Clamp to [0,1]
    nx = fmin(fmax(nx, 0.0f), 1.0f);
    ny = fmin(fmax(ny, 0.0f), 1.0f);
    nz = fmin(fmax(nz, 0.0f), 1.0f);

    morton[gid] = morton3D(nx, ny, nz);
}