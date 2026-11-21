#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL FP_CONTRACT OFF

#include "../shared_structs/projection_gpu_shared.h"
#include "mesh_helpers.cl"

#define SUBPIXEL 256
#define HALF_PIXEL (SUBPIXEL / 2)

typedef struct {
    int x, y;
    float z;
} ScreenVertex;

typedef struct {
    int first, end;
    bool left;
    int x, step;
    long error, stepError, divisor;
} EdgeWalker;

static inline float projectRow(const float *row, float3 p)
{
    return fma(row[0], p.x, fma(row[1], p.y, fma(row[2], p.z, row[3])));
}

static inline float4 project(ProjectionGPU projection, float3 p)
{
    return (float4)(projectRow(projection.rows[0], p), projectRow(projection.rows[1], p),
                    projectRow(projection.rows[2], p), projectRow(projection.rows[3], p));
}

static inline ScreenVertex toScreen(float4 p, int width, int height)
{
    return (ScreenVertex){(int)round(clamp(0.5f * (p.x / p.w + 1.0f), 0.0f, 1.0f) * width * SUBPIXEL),
                          (int)round(clamp(0.5f * (p.y / p.w + 1.0f), 0.0f, 1.0f) * height * SUBPIXEL),
                          clamp(p.z / p.w, 0.0f, 1.0f)};
}

static inline int firstRow(int y)
{
    return (y - HALF_PIXEL + SUBPIXEL - 1) / SUBPIXEL;
}

static inline int floorQuotient(long n, long d, long *remainder)
{
    int q = (int)(n / d);
    *remainder = n % d;
    if (*remainder < 0) { --q; *remainder += d; }
    return q;
}

// Bresenham: represent the intersection as a quotient and a remainder, then
// advance one row with an integer step and an error carry.
static inline EdgeWalker makeEdgeWalker(ScreenVertex a, ScreenVertex b)
{
    EdgeWalker edge;
    edge.left = a.y > b.y;
    if (edge.left) { ScreenVertex tmp = a; a = b; b = tmp; }
    edge.first = firstRow(a.y);
    edge.end = firstRow(b.y);
    if (edge.first == edge.end) return edge;
    const long dx = b.x - a.x, dy = b.y - a.y;
    edge.divisor = dy * SUBPIXEL;
    const long n = (a.x - HALF_PIXEL) * dy + dx * (edge.first * SUBPIXEL + HALF_PIXEL - a.y);
    edge.x = floorQuotient(n, edge.divisor, &edge.error);
    edge.step = floorQuotient(dx * SUBPIXEL, edge.divisor, &edge.stepError);
    return edge;
}

static inline uint shadeFragment(float3 v0, float3 v1, float3 v2)
{
    const float3 n = cross(v1 - v0, v2 - v0);
    const float len = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (!(len > 0.0f)) return 0;
    const uint3 c = convert_uint3(round(255.0f * clamp(0.5f * (n / len) + 0.5f, 0.0f, 1.0f)));
    return c.x | (c.y << 8) | (c.z << 16);
}

static inline void writeFragment(volatile __global ulong *pixel, float depth, uint color)
{
    // cl_khr_int64_base_atomics supplies CAS, not atom_min. Read atomically too.
    const ulong key = ((ulong)as_uint(depth) << 32) | color;
    const ulong empty = ~(ulong)0;
    ulong previous = atom_cmpxchg(pixel, empty, empty);
    while (key < previous) {
        const ulong observed = atom_cmpxchg(pixel, previous, key);
        if (observed == previous) break;
        previous = observed;
    }
}

static inline void rasterizeScreenTriangle(ScreenVertex a, ScreenVertex b, ScreenVertex c,
                                            int width, int height, uint color, __global ulong *framebuffer)
{
    long area = (long)(b.x - a.x) * (c.y - a.y) - (long)(b.y - a.y) * (c.x - a.x);
    if (area == 0) return;
    if (area < 0) { ScreenVertex tmp = b; b = c; c = tmp; area = -area; }
    const float dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / (float)area;
    const float dzdy = ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)) / (float)area;
    EdgeWalker edges[3] = {makeEdgeWalker(a, b), makeEdgeWalker(b, c), makeEdgeWalker(c, a)};
    const int first = firstRow(min(a.y, min(b.y, c.y)));
    const int end = min(height, firstRow(max(a.y, max(b.y, c.y))));
    for (int y = first; y < end; ++y) {
        int left = 0, right = width;
        for (int e = 0; e < 3; ++e) {
            EdgeWalker *edge = &edges[e];
            if (y < edge->first || y >= edge->end) continue;
            const int x = edge->x + (edge->error != 0);
            if (edge->left) left = max(0, x);
            else right = min(width, x);
            edge->x += edge->step;
            edge->error += edge->stepError;
            if (edge->error >= edge->divisor) { ++edge->x; edge->error -= edge->divisor; }
        }
        // Top-left fill rule: [left, right), pixel centers at .5.
        const float rowDepth = fma(dzdy, (float)(y * SUBPIXEL + HALF_PIXEL - a.y), a.z);
        for (int x = left; x < right; ++x) {
            const float z = fma(dzdx, (float)(x * SUBPIXEL + HALF_PIXEL - a.x), rowDepth);
            writeFragment(&framebuffer[(size_t)y * width + x], z > 0.0f ? min(1.0f, z) : 0.0f, color);
        }
    }
}

static inline float planeDistance(float4 p, int plane)
{
    switch (plane) {
        case 0: return p.z;
        case 1: return p.w - p.z;
        case 2: return p.w + p.x;
        case 3: return p.w - p.x;
        case 4: return p.w + p.y;
        default: return p.w - p.y;
    }
}

static inline float4 clipIntersection(float4 a, float4 b, float da, float db)
{
    if (da < 0.0f) {
        float4 tmp = a; a = b; b = tmp;
        float d = da; da = db; db = d;
    }
    return fma((float4)(da / (da - db)), b - a, a);
}

static inline void rasterizeTriangle(float4 a, float4 b, float4 c, int width, int height,
                                      uint color, __global ulong *framebuffer)
{
    uint clipMask = 0;
    for (int plane = 0; plane < 6; ++plane) {
        bool ca = planeDistance(a, plane) < 0.0f;
        bool cb = planeDistance(b, plane) < 0.0f;
        bool cc = planeDistance(c, plane) < 0.0f;
        if (ca && cb && cc) return;
        if (ca || cb || cc) clipMask |= 1u << plane;
    }
    if (!clipMask) {
        rasterizeScreenTriangle(toScreen(a, width, height), toScreen(b, width, height),
                                toScreen(c, width, height), width, height, color, framebuffer);
        return;
    }

    // Homogeneous Sutherland–Hodgman clipping, then triangulate the polygon.
    float4 polygons[2][9] = {{a, b, c}};
    int input = 0, size = 3;
    for (int plane = 0; plane < 6; ++plane) {
        if (!(clipMask & (1u << plane))) continue;
        int count = 0;
        float4 prev = polygons[input][size - 1];
        float dp = planeDistance(prev, plane);
        for (int i = 0; i < size; ++i) {
            const float4 curr = polygons[input][i];
            const float dc = planeDistance(curr, plane);
            if ((dp < 0.0f && dc > 0.0f) || (dp > 0.0f && dc < 0.0f)) {
                polygons[1 - input][count++] = clipIntersection(prev, curr, dp, dc);
            }
            if (dc >= 0.0f) polygons[1 - input][count++] = curr;
            prev = curr;
            dp = dc;
        }
        if (count < 3) return;
        size = count;
        input = 1 - input;
    }
    const ScreenVertex first = toScreen(polygons[input][0], width, height);
    for (int i = 2; i < size; ++i) {
        rasterizeScreenTriangle(first, toScreen(polygons[input][i - 1], width, height),
                                toScreen(polygons[input][i], width, height), width, height, color, framebuffer);
    }
}

__kernel void sw_rasterize_normals(__global const float *vertices, __global const uint *faces,
                                   __global ulong *framebuffer, ProjectionGPU projection,
                                   uint nfaces, int width, int height)
{
    const uint face_id = get_global_id(0);
    if (face_id >= nfaces) return;
    const uint3 face = loadFace(faces, face_id);
    const float3 v0 = loadVertex(vertices, face.x);
    const float3 v1 = loadVertex(vertices, face.y);
    const float3 v2 = loadVertex(vertices, face.z);
    // A flat normal is the same for every fragment of this triangle.
    const uint color = shadeFragment(v0, v1, v2);
    rasterizeTriangle(project(projection, v0), project(projection, v1), project(projection, v2),
                      width, height, color, framebuffer);
}
