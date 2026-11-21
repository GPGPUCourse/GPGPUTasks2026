#include <cuda_runtime.h>

#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>

#include <cstdint>

#include "../kernels.h"
#include "mesh_helpers.cu"

namespace {

constexpr int SUBPIXEL = 256;
constexpr int HALF_PIXEL = SUBPIXEL / 2;

struct ScreenVertex {
    int x, y; // fixed-point screen coordinates, Y points down
    float z;  // post-divide depth, affine in screen space
};

__device__ float projectRow(const float *row, float3 p)
{
    return fmaf(row[0], p.x, fmaf(row[1], p.y, fmaf(row[2], p.z, row[3])));
}

__device__ float4 project(const ProjectionGPU &projection, float3 p)
{
    const auto &r = projection.rows;
    return {projectRow(r[0], p), projectRow(r[1], p), projectRow(r[2], p), projectRow(r[3], p)};
}

__device__ float clamp01(float x) { return x > 0.0f ? fminf(1.0f, x) : 0.0f; }

__device__ uint32_t encodeNormal(float n)
{
    return static_cast<uint32_t>(roundf(255.0f * clamp01(0.5f * n + 0.5f)));
}

__device__ uint32_t shadeFragment(float3 a, float3 b, float3 c)
{
    const float3 u{b.x - a.x, b.y - a.y, b.z - a.z};
    const float3 v{c.x - a.x, c.y - a.y, c.z - a.z};
    const float3 n{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
    const float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (!(length > 0.0f)) return 0;
    return encodeNormal(n.x / length) | (encodeNormal(n.y / length) << 8) | (encodeNormal(n.z / length) << 16);
}

__device__ ScreenVertex toScreen(float4 p, int width, int height)
{
    return {static_cast<int>(roundf(clamp01(0.5f * (p.x / p.w + 1.0f)) * width * SUBPIXEL)),
            static_cast<int>(roundf(clamp01(0.5f * (p.y / p.w + 1.0f)) * height * SUBPIXEL)),
            clamp01(p.z / p.w)};
}

__device__ int firstRow(int y)
{
    return (y - HALF_PIXEL + SUBPIXEL - 1) / SUBPIXEL;
}

// Bresenham: moving down one scanline adds dx/dy using an integer step and an error carry.
struct EdgeWalker {
    int first, end;
    bool left;
    int x, step;
    int64_t error, stepError, divisor;

    __device__ EdgeWalker(ScreenVertex a, ScreenVertex b) : left(a.y > b.y)
    {
        if (left) { const auto tmp = a; a = b; b = tmp; }
        first = firstRow(a.y);
        end = firstRow(b.y);
        if (first == end) return;
        const int64_t dx = b.x - a.x, dy = b.y - a.y;
        divisor = dy * SUBPIXEL;
        const int64_t n = (a.x - HALF_PIXEL) * dy + dx * (first * SUBPIXEL + HALF_PIXEL - a.y);
        x = floorQuotient(n, divisor, error);
        step = floorQuotient(dx * SUBPIXEL, divisor, stepError);
    }

    __device__ static int floorQuotient(int64_t n, int64_t d, int64_t &remainder)
    {
        int q = static_cast<int>(n / d);
        remainder = n % d;
        if (remainder < 0) { --q; remainder += d; }
        return q;
    }

    __device__ int ceilX() const { return x + (error != 0); }

    __device__ void next()
    {
        x += step;
        error += stepError;
        if (error >= divisor) { ++x; error -= divisor; }
    }
};

__device__ void rasterizeScreenTriangle(ScreenVertex a, ScreenVertex b, ScreenVertex c,
                                        int width, int height, uint32_t color, uint64_t *framebuffer)
{
    int64_t area = int64_t(b.x - a.x) * (c.y - a.y) - int64_t(b.y - a.y) * (c.x - a.x);
    if (area == 0) return;
    if (area < 0) { const auto tmp = b; b = c; c = tmp; area = -area; }
    const float dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / float(area);
    const float dzdy = ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)) / float(area);
    EdgeWalker edges[] = {{a, b}, {b, c}, {c, a}};
    const int first = firstRow(min(a.y, min(b.y, c.y)));
    const int end = min(height, firstRow(max(a.y, max(b.y, c.y))));
    for (int y = first; y < end; ++y) {
        int left = 0, right = width;
        for (auto &edge : edges) {
            if (y < edge.first || y >= edge.end) continue;
            if (edge.left) left = max(0, edge.ceilX());
            else right = min(width, edge.ceilX());
            edge.next();
        }
        // Top-left fill rule: include the left and top edges, exclude right and bottom.
        const float row_depth = fmaf(dzdy, float(y * SUBPIXEL + HALF_PIXEL - a.y), a.z);
        for (int x = left; x < right; ++x) {
            const float depth = clamp01(fmaf(dzdx, float(x * SUBPIXEL + HALF_PIXEL - a.x), row_depth));
            // Nonnegative float bits sort as integers. Update depth and RGB together.
            const auto fragment = (static_cast<unsigned long long>(__float_as_uint(depth)) << 32) | color;
            atomicMin((unsigned long long*) &framebuffer[size_t(y) * width + x], fragment);
        }
    }
}

__device__ float planeDistance(float4 p, int plane)
{
    switch (plane) {
        case 0: return p.z;         // near
        case 1: return p.w - p.z;   // far
        case 2: return p.w + p.x;   // left
        case 3: return p.w - p.x;   // right
        case 4: return p.w + p.y;   // top
        default: return p.w - p.y;  // bottom
    }
}

__device__ float4 clipIntersection(float4 a, float4 b, float da, float db)
{
    // Interpolate from the inside endpoint, so shared edges clip identically.
    if (da < 0.0f) {
        const auto tmp = a; a = b; b = tmp;
        const float d = da; da = db; db = d;
    }
    const float t = da / (da - db);
    return {fmaf(t, b.x - a.x, a.x), fmaf(t, b.y - a.y, a.y),
            fmaf(t, b.z - a.z, a.z), fmaf(t, b.w - a.w, a.w)};
}

__device__ void rasterizeTriangle(float4 a, float4 b, float4 c, int width, int height,
                                  uint32_t color, uint64_t *framebuffer)
{
    unsigned clip_mask = 0;
    for (int plane = 0; plane < 6; ++plane) {
        const bool ca = planeDistance(a, plane) < 0.0f;
        const bool cb = planeDistance(b, plane) < 0.0f;
        const bool cc = planeDistance(c, plane) < 0.0f;
        if (ca && cb && cc) return;
        if (ca || cb || cc) clip_mask |= 1u << plane;
    }
    if (!clip_mask) {
        rasterizeScreenTriangle(toScreen(a, width, height), toScreen(b, width, height), toScreen(c, width, height),
                                width, height, color, framebuffer);
        return;
    }

    // Sutherland–Hodgman clipping: each plane adds at most one vertex.
    float4 polygons[2][9] = {{a, b, c}};
    int input = 0, size = 3;
    for (int plane = 0; plane < 6; ++plane) {
        if (!(clip_mask & (1u << plane))) continue;
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

    const auto *polygon = polygons[input];
    const ScreenVertex first = toScreen(polygon[0], width, height);
    for (int i = 2; i < size; ++i) {
        rasterizeScreenTriangle(first, toScreen(polygon[i - 1], width, height), toScreen(polygon[i], width, height),
                                width, height, color, framebuffer);
    }
}

__global__ void sw_rasterize_normals(const float *vertices, const unsigned int *faces,
                                     uint64_t *framebuffer, ProjectionGPU projection,
                                     unsigned int nfaces, int width, int height)
{
    const unsigned int face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id >= nfaces) return;
    const uint3 face = loadFace(faces, face_id);
    const float3 v0 = loadVertex(vertices, face.x);
    const float3 v1 = loadVertex(vertices, face.y);
    const float3 v2 = loadVertex(vertices, face.z);
    // A flat normal is the same for every fragment of this triangle.
    const uint32_t color = shadeFragment(v0, v1, v2);
    rasterizeTriangle(project(projection, v0), project(projection, v1), project(projection, v2),
                      width, height, color, framebuffer);
}

} // namespace

namespace cuda {

void sw_rasterize_normals(const gpu::WorkSize &work_size, const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u &faces, const gpu::gpu_mem_64u &framebuffer,
    const ProjectionGPU &projection, unsigned int nfaces, int width, int height)
{
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();
    ::sw_rasterize_normals<<<work_size.cuGridSize(), work_size.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), framebuffer.cuptr(), projection, nfaces, width, height);
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
