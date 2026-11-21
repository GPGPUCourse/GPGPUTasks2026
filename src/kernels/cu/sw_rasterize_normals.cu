#include <cuda_runtime.h>

#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

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

    __device__ EdgeWalker(ScreenVertex a, ScreenVertex b, int start = 0, int stop = INT_MAX, bool fast = false)
        : left(a.y > b.y)
    {
        if (left) { const auto tmp = a; a = b; b = tmp; }
        first = max(start, firstRow(a.y));
        end = min(stop, firstRow(b.y));
        if (first >= end) return;
        const int64_t dx = b.x - a.x, dy = b.y - a.y;
        divisor = dy * SUBPIXEL;
        const int64_t n = (a.x - HALF_PIXEL) * dy + dx * (first * SUBPIXEL + HALF_PIXEL - a.y);
        if (fast) {
            x = fastFloorQuotient(n, divisor, error);
            if (end - first > 1) step = fastFloorQuotient(dx * SUBPIXEL, divisor, stepError);
            else { step = 0; stepError = 0; }
        } else {
            x = floorQuotient(n, divisor, error);
            step = floorQuotient(dx * SUBPIXEL, divisor, stepError);
        }
    }

    __device__ static int fastFloorQuotient(int64_t n, int64_t d, int64_t &remainder)
    {
        // Estimate the quotient, then correct in integer arithmetic to preserve coverage.
        int q = __float2int_rd(__fdividef(float(n), float(d)));
        remainder = n - int64_t(q) * d;
        while (remainder < 0) { --q; remainder += d; }
        while (remainder >= d) { ++q; remainder -= d; }
        return q;
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

template<class Emit>
__device__ void clipAndProject(float4 a, float4 b, float4 c, int width, int height, Emit emit)
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
        emit(toScreen(a, width, height), toScreen(b, width, height), toScreen(c, width, height));
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
        emit(first, toScreen(polygon[i - 1], width, height), toScreen(polygon[i], width, height));
    }
}

struct DrawGlobal {
    int width, height;
    uint32_t color;
    uint64_t *framebuffer;

    __device__ void operator()(ScreenVertex a, ScreenVertex b, ScreenVertex c) const
    {
        rasterizeScreenTriangle(a, b, c, width, height, color, framebuffer);
    }
};

__device__ void rasterizeFace(const float *vertices, const unsigned int *faces, uint64_t *framebuffer,
                              const ProjectionGPU &projection, unsigned int face_id, int width, int height)
{
    const uint3 face = loadFace(faces, face_id);
    const float3 v0 = loadVertex(vertices, face.x);
    const float3 v1 = loadVertex(vertices, face.y);
    const float3 v2 = loadVertex(vertices, face.z);
    // A flat normal is the same for every fragment of this triangle.
    const uint32_t color = shadeFragment(v0, v1, v2);
    clipAndProject(project(projection, v0), project(projection, v1), project(projection, v2),
                   width, height, DrawGlobal{width, height, color, framebuffer});
}

__global__ void sw_rasterize_normals(const float *vertices, const unsigned int *faces,
                                     uint64_t *framebuffer, ProjectionGPU projection,
                                     unsigned int nfaces, int width, int height)
{
    const unsigned int face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id < nfaces) rasterizeFace(vertices, faces, framebuffer, projection, face_id, width, height);
}

#include "tiled_rasterizer.cuh"

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

struct TiledRasterizer::Impl {
    RasterGrid grid{};
    RasterBuffers buffers{};
    unsigned faceCount, smCount;
    bool enabled = false;
    ScratchArray<SetupTriangle> triangles;
    ScratchArray<unsigned> binRefs, tileRefs, activeTiles, tileParts;
    ScratchArray<Range> bins, tiles;
    ScratchArray<CoarseTask> tasks;
    ScratchArray<RasterCounters> counters;

    Impl(unsigned nfaces, int width, int height) : faceCount(nfaces)
    {
        int device;
        cudaDeviceProp prop;
        CUDA_SAFE_CALL(cudaGetDevice(&device));
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
        smCount = prop.multiProcessorCount;
        constexpr int maxSide = INT_MAX / SUBPIXEL - BIN_SIZE;
        if (width <= 0 || height <= 0 || width > maxSide || height > maxSide || nfaces > UINT_MAX / 7u) return;
        grid = {width, height, div_ceil(width, BIN_SIZE), div_ceil(height, BIN_SIZE),
                div_ceil(width, TILE_SIZE), div_ceil(height, TILE_SIZE)};
        const size_t numBins = size_t(grid.binsX) * grid.binsY;
        if (numBins * sizeof(unsigned) > prop.sharedMemPerBlock || size_t(grid.tilesX) * grid.tilesY > INT_MAX) return;

        const unsigned triCapacity = nfaces + nfaces / 4 + 1024;
        const unsigned binCapacity = std::max(2u * nfaces, 32u * static_cast<unsigned>(numBins));
        const unsigned tileCapacity = static_cast<unsigned>(std::min<size_t>(UINT_MAX,
            std::max<size_t>(4ull * nfaces, 16ull * grid.numTiles())));
        triangles.allocate(triCapacity);
        binRefs.allocate(binCapacity);
        tileRefs.allocate(tileCapacity);
        activeTiles.allocate(grid.numTiles());
        tileParts.allocate(grid.numTiles());
        bins.allocate(numBins * smCount);
        tiles.allocate(size_t(grid.numTiles()) * MAX_BIN_PARTS);
        tasks.allocate(numBins * MAX_BIN_PARTS);
        counters.allocate(1);
        buffers = {triangles.ptr, binRefs.ptr, tileRefs.ptr, activeTiles.ptr, tileParts.ptr,
                   bins.ptr, tiles.ptr, tasks.ptr, counters.ptr, triCapacity, binCapacity, tileCapacity};
        enabled = true;
    }
};

TiledRasterizer::TiledRasterizer(unsigned nfaces, int width, int height)
    : impl(std::make_unique<Impl>(nfaces, width, height))
{
    impl->grid.width = width;
    impl->grid.height = height;
}

TiledRasterizer::~TiledRasterizer() = default;

void TiledRasterizer::draw(const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
                           const gpu::gpu_mem_64u &framebuffer, const ProjectionGPU &projection)
{
    if (!impl->faceCount) return;
    const auto &s = *impl;
    if (!s.enabled) {
        sw_rasterize_normals(gpu::WorkSize(BIN_THREADS, s.faceCount), vertices, faces, framebuffer,
                             projection, s.faceCount, s.grid.width, s.grid.height);
        return;
    }
    gpu::Context context;
    cudaStream_t stream = context.cudaStream();
    CUDA_SAFE_CALL(cudaMemsetAsync(s.buffers.counters, 0, sizeof(RasterCounters), stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(s.buffers.tileParts, 0, s.grid.numTiles() * sizeof(unsigned), stream));
    triangleSetup<<<div_ceil(s.faceCount, unsigned(BIN_THREADS)), BIN_THREADS, 0, stream>>>(
        vertices.cuptr(), faces.cuptr(), projection, s.faceCount, s.grid, s.buffers);
    binRasterizer<<<s.smCount, BIN_THREADS, s.grid.numBins() * sizeof(unsigned), stream>>>(s.grid, s.buffers);
    makeCoarseTasks<<<div_ceil(s.grid.numBins(), BIN_THREADS / 32), BIN_THREADS, 0, stream>>>(s.grid, s.buffers, s.smCount);
    coarseRasterizer<<<2 * s.smCount, COARSE_THREADS, 0, stream>>>(s.grid, s.buffers, s.smCount);
    fineRasterizer<<<6 * s.smCount, FINE_THREADS, 0, stream>>>(s.grid, s.buffers, framebuffer.cuptr());
    rasterizeOverflow<<<s.smCount, BIN_THREADS, 0, stream>>>(vertices.cuptr(), faces.cuptr(), framebuffer.cuptr(),
        projection, s.faceCount, s.grid, s.buffers.counters);
    CUDA_CHECK_KERNEL_SYNC(stream);
}

TiledRasterizer::Stats TiledRasterizer::stats() const
{
    if (!impl->enabled) return {0, 0, 0, 0, true};
    if (!impl->faceCount) return {};
    RasterCounters info;
    CUDA_SAFE_CALL(cudaMemcpy(&info, impl->buffers.counters, sizeof(info), cudaMemcpyDeviceToHost));
    return {info.numTriangles, info.numActiveTiles, info.numBinRefs, info.numTileRefs, info.overflow != 0};
}

} // namespace cuda
