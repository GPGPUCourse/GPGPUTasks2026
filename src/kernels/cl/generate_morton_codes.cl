#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

// Spread the lower 10 bits of v so there are 2 zero bits between each bit
static inline uint expandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// x, y, z in [0, 1]
static inline uint morton3D(float x, float y, float z)
{
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    const uint xx = expandBits((uint)x);
    const uint yy = expandBits((uint)y);
    const uint zz = expandBits((uint)z);
    return (xx << 2) | (yy << 1) | zz;
}

// One work-item per triangle.
// triIndices is initialized to identity; it becomes the "values" array for the radix sort.
__kernel void generate_morton_codes(
    __global const float* centroids, // 3 floats per triangle
    __global const float* centroidBounds, // [0..2] = min, [3..5] = max
    __global uint* mortonCodes,
    __global uint* triIndices,
    uint nfaces)
{
    const uint gid = get_global_id(0);
    if (gid >= nfaces)
        return;

    const float3 cMin = vload3(0, centroidBounds);
    const float3 cMax = vload3(1, centroidBounds);
    const float3 ext = fmax(cMax - cMin, (float3)(1e-9f)); // same eps as the CPU code

    const float3 n = clamp((vload3(gid, centroids) - cMin) / ext, 0.0f, 1.0f);

    mortonCodes[gid] = morton3D(n.x, n.y, n.z);
    triIndices[gid] = gid;
}