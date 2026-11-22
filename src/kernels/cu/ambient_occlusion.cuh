#pragma once
// https://github.com/GPGPUCourse/GPGPUTasks2026/blob/c0f3bf6f9da91a9a3f3e3596fcbd2e981d265339/src/kernels/cu/ray_tracing_render_brute_force.cu#L37
__device__ inline void make_basis(const float3& n, float3& t, float3& b) {
    float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.f,0.f,1.f) : make_float3(0.f,1.f,0.f);
    t = normalize_f3(cross_f3(up, n));
    b = cross_f3(n, t);
}

// https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-17-ambient-occlusion
// https://github.com/syoyo/aobench/blob/590a98d00a72526a7f3bd9b5065213dd00fd94fe/ao.c#L154-L210
template<class Occluded>
static __device__ float ambient_occlusion(const float* vertices, const unsigned* faces,
    float3 ray_origin, float3 ray_direction, int faceIdBest, float tBest, unsigned idx, Occluded occluded)
{
    float ao = 1.0f;
    if (faceIdBest >= 0) {
        uint3 f = loadFace(faces, faceIdBest);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
        float3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        float3 n  = normalize_f3(cross_f3(e1, e2));

        if (n.x*ray_direction.x + n.y*ray_direction.y + n.z*ray_direction.z > 0.0f)
            n = make_float3(-n.x, -n.y, -n.z);

        float3 P  = {ray_origin.x + tBest*ray_direction.x,
            ray_origin.y + tBest*ray_direction.y,
            ray_origin.z + tBest*ray_direction.z};

        float scale = fmaxf(fmaxf(length_f3(e1), length_f3(e2)),
            length_f3(make_float3(c.x-a.x, c.y-a.y, c.z-a.z)));
        float eps   = 1e-3f * fmaxf(1.0f, scale);
        float3 Po   = {P.x + n.x*eps, P.y + n.y*eps, P.z + n.z*eps};

        float3 T, B;
        make_basis(n, T, B);
        union {
            float f32;
            uint32_t u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint32_t rng = 0x9E3779B9u ^ idx ^ tBestUnion.u32;

        // https://github.com/GPGPUCourse/GPGPUTasks2026/blob/c0f3bf6f9da91a9a3f3e3596fcbd2e981d265339/src/kernels/cu/ray_tracing_render_brute_force.cu#L84-L139
        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            float u1 = random01(rng);
            float u2 = random01(rng);
            float z  = u1;
            float phi = 6.28318530718f * u2;
            float r  = sqrtf(fmaxf(0.f, 1.f - z*z));
            float3 d_local = make_float3(r * cosf(phi), r * sinf(phi), z);

            float3 d = make_float3(
                T.x*d_local.x + B.x*d_local.y + n.x*d_local.z,
                T.y*d_local.x + B.y*d_local.y + n.y*d_local.z,
                T.z*d_local.x + B.z*d_local.y + n.z*d_local.z
            );

            if (occluded(Po, d, faceIdBest))
                ++hits;
        }
        ao = 1.0f - (float)hits / (float)AO_SAMPLES;
    }
    return ao;
}
