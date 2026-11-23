#include "../defines.h"

static inline uint lbvh_expand_bits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void lbvh_morton_codes(
    __global const float* centroids,
    __global const float* scene_bounds,
    __global uint* codes,
    __global uint* triangle_ids,
    uint nfaces)
{
    const uint i = get_global_id(0);
    if (i >= nfaces)
        return;

    const float3 lo = (float3)(scene_bounds[0], scene_bounds[1], scene_bounds[2]);
    const float3 hi = (float3)(scene_bounds[3], scene_bounds[4], scene_bounds[5]);
    const float3 c = (float3)(centroids[3 * i], centroids[3 * i + 1], centroids[3 * i + 2]);
    const float3 extent = fmax(hi - lo, (float3)(1e-9f, 1e-9f, 1e-9f));
    const float3 normalized = clamp((c - lo) / extent, 0.0f, 1.0f);
    const uint x = min((uint)(normalized.x * 1024.0f), 1023u);
    const uint y = min((uint)(normalized.y * 1024.0f), 1023u);
    const uint z = min((uint)(normalized.z * 1024.0f), 1023u);
    codes[i] = (lbvh_expand_bits(x) << 2) | (lbvh_expand_bits(y) << 1) | lbvh_expand_bits(z);
    triangle_ids[i] = i;
}
