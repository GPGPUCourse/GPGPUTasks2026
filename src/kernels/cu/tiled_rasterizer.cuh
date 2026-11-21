// Included by sw_rasterize_normals.cu: shares its projection, clipping and Bresenham rules.

constexpr int TILE_SIZE = 8;
constexpr int BIN_SIZE = 128;
constexpr int TILES_PER_BIN = BIN_SIZE / TILE_SIZE;
constexpr int BIN_THREADS = 256;
constexpr int COARSE_THREADS = TILES_PER_BIN * TILES_PER_BIN;
constexpr int WARP_SIZE = 32;
constexpr int FINE_WARPS = 4;
constexpr int FINE_THREADS = FINE_WARPS * WARP_SIZE;
constexpr unsigned FULL_WARP = 0xffffffffu;
constexpr unsigned MAX_BIN_PARTS = 16;
constexpr unsigned BIN_REFS_PER_PART = 4096;
using PixelKey = unsigned long long;

struct SetupTriangle {
    ScreenVertex a, b, c; // positive winding in the framebuffer coordinate system
    uint32_t color;
};

struct Range {
    unsigned begin, count;
};

struct RasterGrid {
    int width, height, binsX, binsY, tilesX, tilesY;

    __host__ __device__ int numBins() const { return binsX * binsY; }
    __host__ __device__ int numTiles() const { return tilesX * tilesY; }
};

struct RasterCounters {
    unsigned numTriangles, nextTask, numTasks, numActiveTiles, overflow;
    unsigned long long numBinRefs, numTileRefs;
};

struct CoarseTask {
    unsigned bin, part, parts;
};

struct RasterBuffers {
    SetupTriangle *triangles;
    unsigned *binRefs, *tileRefs, *activeTiles, *tileParts;
    Range *bins, *tiles;
    CoarseTask *tasks;
    RasterCounters *counters;
    unsigned triangleCapacity, binCapacity, tileCapacity;
};

struct PixelBounds {
    int x0, y0, x1, y1; // half-open bounds, in pixel coordinates
};

__device__ PixelBounds pixelBounds(ScreenVertex a, ScreenVertex b, ScreenVertex c)
{
    return {firstRow(min(a.x, min(b.x, c.x))), firstRow(min(a.y, min(b.y, c.y))),
            firstRow(max(a.x, max(b.x, c.x))), firstRow(max(a.y, max(b.y, c.y)))};
}

__device__ int64_t edgeValue(ScreenVertex a, ScreenVertex b, int x, int y)
{
    return int64_t(b.x - a.x) * (y - a.y) - int64_t(b.y - a.y) * (x - a.x);
}

__device__ bool coversSample(ScreenVertex a, ScreenVertex b, ScreenVertex c, int x, int y)
{
    const int sampleX = x * SUBPIXEL + HALF_PIXEL, sampleY = y * SUBPIXEL + HALF_PIXEL;
    const auto edge = [=](ScreenVertex u, ScreenVertex v) {
        const int64_t e = edgeValue(u, v, sampleX, sampleY);
        return e > 0 || (e == 0 && (v.y < u.y || (v.y == u.y && v.x > u.x)));
    };
    return edge(a, b) && edge(b, c) && edge(c, a);
}

struct EmitTriangle {
    RasterBuffers buffers;
    float3 v0, v1, v2;

    __device__ void operator()(ScreenVertex a, ScreenVertex b, ScreenVertex c) const
    {
        const auto bounds = pixelBounds(a, b, c);
        if (bounds.x0 >= bounds.x1 || bounds.y0 >= bounds.y1) return;
        const int64_t area = edgeValue(a, b, c.x, c.y);
        if (area == 0) return;
        if (area < 0) { const auto tmp = b; b = c; c = tmp; }
        const int dx = bounds.x1 - bounds.x0, dy = bounds.y1 - bounds.y0;
        if (dx + dy <= 3 && !coversSample(a, b, c, bounds.x0, bounds.y0) &&
            (dx + dy == 2 || !coversSample(a, b, c, bounds.x1 - 1, bounds.y1 - 1))) return;

        const uint32_t color = shadeFragment(v0, v1, v2);
        // Compact surviving triangles, using one allocation per participating group of lanes.
        const unsigned lanes = __activemask();
        const int leader = __ffs(lanes) - 1;
        const int lane = threadIdx.x & 31;
        unsigned base = 0;
        if (lane == leader) base = atomicAdd(&buffers.counters->numTriangles, __popc(lanes));
        base = __shfl_sync(lanes, base, leader);
        const unsigned index = base + __popc(lanes & ((1u << lane) - 1u));
        if (index < buffers.triangleCapacity) buffers.triangles[index] = {a, b, c, color};
        else if (index == buffers.triangleCapacity) atomicExch(&buffers.counters->overflow, 1u);
    }
};

__device__ unsigned clipOutcode(float4 p)
{
    return (unsigned(p.z < 0.0f) << 0) | (unsigned(p.w - p.z < 0.0f) << 1) |
           (unsigned(p.w + p.x < 0.0f) << 2) | (unsigned(p.w - p.x < 0.0f) << 3) |
           (unsigned(p.w + p.y < 0.0f) << 4) | (unsigned(p.w - p.y < 0.0f) << 5);
}

// The slow path may allocate a clipping polygon, but it only initializes it for
// triangles that actually cross a plane. Inlining can move those stores before culling.
__device__ __noinline__ void setupClipped(float4 pa, float4 pb, float4 pc, unsigned mask,
                                          int width, int height, EmitTriangle emit)
{
    clipPartial(pa, pb, pc, mask, width, height, emit);
}

__global__ void triangleSetup(const float *vertices, const unsigned *faces, ProjectionGPU projection,
                              unsigned nfaces, RasterGrid grid, RasterBuffers buffers)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;
    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x), b = loadVertex(vertices, face.y), c = loadVertex(vertices, face.z);
    const float4 pa = project(projection, a), pb = project(projection, b), pc = project(projection, c);
    const unsigned ma = clipOutcode(pa), mb = clipOutcode(pb), mc = clipOutcode(pc);
    if (ma & mb & mc) return;
    const unsigned mask = ma | mb | mc;
    const EmitTriangle emit{buffers, a, b, c};
    if (mask) {
        setupClipped(pa, pb, pc, mask, grid.width, grid.height, emit);
    } else {
        emit(toScreen(pa, grid.width, grid.height), toScreen(pb, grid.width, grid.height),
             toScreen(pc, grid.width, grid.height));
    }
}

// Find the largest edge value on a rectangle of sample centers. Including edge ties here
// is conservative: the top-left rule is applied by Fine, not by the queue builders.
__device__ bool edgeMayCover(ScreenVertex a, ScreenVertex b, int x, int y, int size)
{
    const int sampleX = (x + (b.y < a.y ? size - 1 : 0)) * SUBPIXEL + HALF_PIXEL;
    const int sampleY = (y + (b.x > a.x ? size - 1 : 0)) * SUBPIXEL + HALF_PIXEL;
    return edgeValue(a, b, sampleX, sampleY) >= 0;
}

__device__ bool mayCover(const SetupTriangle &tri, int x, int y, int size)
{
    return edgeMayCover(tri.a, tri.b, x, y, size) && edgeMayCover(tri.b, tri.c, x, y, size) &&
           edgeMayCover(tri.c, tri.a, x, y, size);
}

template<bool Write>
__device__ void binPass(RasterGrid grid, RasterBuffers buffers, unsigned *positions)
{
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < buffers.counters->numTriangles; i += stride) {
        const auto tri = buffers.triangles[i];
        const auto bounds = pixelBounds(tri.a, tri.b, tri.c);
        const int x0 = bounds.x0 / BIN_SIZE, x1 = (bounds.x1 - 1) / BIN_SIZE;
        const int y0 = bounds.y0 / BIN_SIZE, y1 = (bounds.y1 - 1) / BIN_SIZE;
        const bool testCoverage = (x1 - x0 > 1 || y1 - y0 > 1);
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                if (testCoverage && !mayCover(tri, x * BIN_SIZE, y * BIN_SIZE, BIN_SIZE)) continue;
                const unsigned pos = atomicAdd(positions + y * grid.binsX + x, 1u);
                if (Write) buffers.binRefs[pos] = i;
            }
        }
    }
}

// One persistent CTA per shard (grid sized to the number of SMs). Each CTA owns its
// entire set of bin queues. Only the allocation of the backing span is global/atomic.
__global__ void binRasterizer(RasterGrid grid, RasterBuffers buffers)
{
    if (__syncthreads_or(threadIdx.x == 0 && atomicAdd(&buffers.counters->overflow, 0u))) return;
    extern __shared__ unsigned positions[];
    __shared__ bool fits;
    for (int bin = threadIdx.x; bin < grid.numBins(); bin += blockDim.x) positions[bin] = 0;
    __syncthreads();
    binPass<false>(grid, buffers, positions);
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long total = 0;
        for (int bin = 0; bin < grid.numBins(); ++bin) total += positions[bin];
        auto base = total ? atomicAdd(&buffers.counters->numBinRefs, total) : 0ull;
        fits = (base + total <= buffers.binCapacity);
        if (fits) {
            for (int bin = 0; bin < grid.numBins(); ++bin) {
                const unsigned count = positions[bin];
                buffers.bins[bin * gridDim.x + blockIdx.x] = {static_cast<unsigned>(base), count};
                positions[bin] = static_cast<unsigned>(base);
                base += count;
            }
        } else {
            atomicExch(&buffers.counters->overflow, 1u);
        }
    }
    __syncthreads();
    if (fits) binPass<true>(grid, buffers, positions);
}

// Split hot bins across CTAs by input shard. Each (tile, part) has a single writer;
// the Fine CTA still owns all pixels in the tile. Empty bins don't generate jobs.
__global__ void makeCoarseTasks(RasterGrid grid, RasterBuffers buffers, unsigned shards)
{
    if (buffers.counters->overflow) return;
    const unsigned bin = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (bin >= grid.numBins()) return;
    const unsigned lane = threadIdx.x & 31;
    unsigned count = 0;
    for (unsigned shard = lane; shard < shards; shard += 32) count += buffers.bins[bin * shards + shard].count;
    for (int step = 16; step != 0; step >>= 1) count += __shfl_down_sync(0xffffffffu, count, step);
    count = __shfl_sync(0xffffffffu, count, 0);
    if (!count) return;
    const unsigned parts = min(MAX_BIN_PARTS, min(shards, 1 + (count - 1) / BIN_REFS_PER_PART));
    unsigned first = 0;
    if (lane == 0) first = atomicAdd(&buffers.counters->numTasks, parts);
    first = __shfl_sync(0xffffffffu, first, 0);
    if (lane < parts) buffers.tasks[first + lane] = {bin, lane, parts};
}

template<bool Write>
__device__ void coarsePass(RasterGrid grid, RasterBuffers buffers, CoarseTask task, unsigned shards, unsigned *positions)
{
    const int binTileX = (task.bin % grid.binsX) * TILES_PER_BIN;
    const int binTileY = (task.bin / grid.binsX) * TILES_PER_BIN;
    const unsigned lane = threadIdx.x & 31;
    const unsigned shardBegin = shards * task.part / task.parts, shardEnd = shards * (task.part + 1) / task.parts;
    for (unsigned shard = shardBegin + threadIdx.x / 32; shard < shardEnd; shard += COARSE_THREADS / 32) {
        const Range refs = buffers.bins[task.bin * shards + shard];
        for (unsigned j = lane; j < refs.count; j += 32) {
            const unsigned triId = buffers.binRefs[refs.begin + j];
            const auto tri = buffers.triangles[triId];
            const auto bounds = pixelBounds(tri.a, tri.b, tri.c);
            const int x0 = max(0, bounds.x0 / TILE_SIZE - binTileX);
            const int y0 = max(0, bounds.y0 / TILE_SIZE - binTileY);
            const int x1 = min(TILES_PER_BIN - 1, (bounds.x1 - 1) / TILE_SIZE - binTileX);
            const int y1 = min(TILES_PER_BIN - 1, (bounds.y1 - 1) / TILE_SIZE - binTileY);
            const bool testCoverage = (x1 != x0 || y1 != y0);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    if (testCoverage && !mayCover(tri, (binTileX + x) * TILE_SIZE, (binTileY + y) * TILE_SIZE, TILE_SIZE)) continue;
                    const unsigned pos = atomicAdd(positions + y * TILES_PER_BIN + x, 1u);
                    if (Write) buffers.tileRefs[pos] = triId;
                }
            }
        }
    }
}

// Persistent CTAs pick up several bin jobs. Counts/cursors are private to the CTA;
// no order-preserving merge of tile list parts is needed.
__global__ void coarseRasterizer(RasterGrid grid, RasterBuffers buffers, unsigned shards)
{
    if (__syncthreads_or(threadIdx.x == 0 && atomicAdd(&buffers.counters->overflow, 0u))) return;
    __shared__ unsigned positions[COARSE_THREADS], taskId;
    __shared__ bool fits;
    const int lane = threadIdx.x;
    for (;;) {
        if (lane == 0) taskId = atomicAdd(&buffers.counters->nextTask, 1u);
        positions[lane] = 0;
        __syncthreads();
        if (taskId >= buffers.counters->numTasks) return;
        const CoarseTask task = buffers.tasks[taskId];
        coarsePass<false>(grid, buffers, task, shards, positions);
        __syncthreads();

        const unsigned count = positions[lane];
        __syncthreads(); // Capture every count before thread 0 replaces counts with cursors.
        if (lane == 0) {
            unsigned long long total = 0;
            for (int i = 0; i < COARSE_THREADS; ++i) total += positions[i];
            auto base = total ? atomicAdd(&buffers.counters->numTileRefs, total) : 0ull;
            fits = (base + total <= buffers.tileCapacity);
            if (fits) {
                for (int i = 0; i < COARSE_THREADS; ++i) {
                    const unsigned n = positions[i];
                    positions[i] = static_cast<unsigned>(base);
                    base += n;
                }
            } else {
                atomicExch(&buffers.counters->overflow, 1u);
            }
        }
        __syncthreads();
        if (!fits) return;
        if (count != 0) {
            const int tileX = (task.bin % grid.binsX) * TILES_PER_BIN + lane % TILES_PER_BIN;
            const int tileY = (task.bin / grid.binsX) * TILES_PER_BIN + lane / TILES_PER_BIN;
            const int tile = tileY * grid.tilesX + tileX;
            buffers.tiles[size_t(tile) * MAX_BIN_PARTS + task.part] = {positions[lane], count};
            if (!atomicOr(buffers.tileParts + tile, 1u << task.part)) {
                const unsigned active = atomicAdd(&buffers.counters->numActiveTiles, 1u);
                buffers.activeTiles[active] = tile;
            }
        }
        __syncthreads();
        coarsePass<true>(grid, buffers, task, shards, positions);
        __syncthreads();
    }
}

#include "fine_coverage.cuh"

struct FineTriangle {
    Coverage coverage;
    unsigned first;
    float z, dzdx, dzdy;
    int x, y;
    uint32_t color;
};

__device__ void setupFragmentPlane(FineTriangle &out, const SetupTriangle &tri, int x0, int y0)
{
    const auto a = tri.a, b = tri.b, c = tri.c;
    const float area = float(edgeValue(a, b, c.x, c.y));
    out.dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / area;
    out.dzdy = ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)) / area;
    out.x = x0 * SUBPIXEL + HALF_PIXEL - a.x;
    out.y = y0 * SUBPIXEL + HALF_PIXEL - a.y;
    out.z = a.z;
    out.color = tri.color;
}

__device__ void shadeBatch(FineTriangle triangle, unsigned lane, FineTriangle *batch, PixelKey *pixels)
{
    const unsigned count = __popcll(triangle.coverage);
    const unsigned nonempty = __ballot_sync(FULL_WARP, count != 0);
    if (!nonempty) return;
    unsigned end = count;
    #pragma unroll
    for (int step = 1; step < WARP_SIZE; step *= 2) {
        const unsigned prev = __shfl_up_sync(FULL_WARP, end, step);
        if (lane >= step) end += prev;
    }
    const unsigned total = __shfl_sync(FULL_WARP, end, WARP_SIZE - 1);
    triangle.first = end - count;
    if (count) batch[lane] = triangle;
    __syncwarp();

    // Each bit marks a triangle's first fragment in the current group of 32.
    // Its prefix popcount selects the source lane among the nonempty triangles.
    unsigned started = 0;
    const unsigned throughLane = 0xffffffffu >> (31 - lane);
    for (unsigned base = 0; base < total; base += WARP_SIZE) {
        const unsigned offset = triangle.first - base;
        unsigned starts = count && offset < WARP_SIZE ? 1u << offset : 0;
#if __CUDA_ARCH__ >= 800
        starts = __reduce_or_sync(FULL_WARP, starts);
#else
        #pragma unroll
        for (int step = 16; step > 0; step >>= 1) starts |= __shfl_xor_sync(FULL_WARP, starts, step);
#endif
        if (base + lane < total) {
            const unsigned owner = __fns(nonempty, 0, started + __popc(starts & throughLane));
            const FineTriangle info = batch[owner];
            const unsigned pixel = nthPixel(info.coverage, base + lane - info.first);
            const int x = int(pixel & 7), y = int(pixel >> 3);
            const float rowDepth = fmaf(info.dzdy, float(info.y + y * SUBPIXEL), info.z);
            const float z = clamp01(fmaf(info.dzdx, float(info.x + x * SUBPIXEL), rowDepth));
            atomicMin(pixels + pixel, (PixelKey(__float_as_uint(z)) << 32) | info.color);
        }
        started += __popc(starts);
    }
    __syncwarp(); // All reads from batch[] are done before any lane can reuse it.
}

__global__ void fineRasterizer(RasterGrid grid, RasterBuffers buffers, uint64_t *framebuffer)
{
    if (buffers.counters->overflow) return;
    __shared__ PixelKey warpPixels[FINE_WARPS][TILE_SIZE * TILE_SIZE];
    __shared__ FineTriangle warpBatch[FINE_WARPS][WARP_SIZE];
    const unsigned warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    auto *pixels = warpPixels[warp];
    for (unsigned i = blockIdx.x * FINE_WARPS + warp; i < buffers.counters->numActiveTiles;
         i += gridDim.x * FINE_WARPS) {
        const int tile = buffers.activeTiles[i];
        const int x0 = (tile % grid.tilesX) * TILE_SIZE, y0 = (tile / grid.tilesX) * TILE_SIZE;
        const unsigned width = min(TILE_SIZE, grid.width - x0), height = min(TILE_SIZE, grid.height - y0);
        const Coverage valid = (COLUMN_BITS * ((1u << width) - 1u)) & (~0ull >> (8 * (8 - height)));
        #pragma unroll
        for (unsigned slot = lane; slot < 64; slot += WARP_SIZE) {
            const int x = x0 + int(slot & 7), y = y0 + int(slot >> 3);
            pixels[slot] = ((valid >> slot) & 1) ? framebuffer[size_t(y) * grid.width + x] : ~PixelKey{0};
        }
        __syncwarp();

        unsigned parts = buffers.tileParts[tile], ref = 0, left = 0;
        while (parts || left) {
            unsigned filled = 0, triId = 0;
            do {
                if (!left) {
                    const int part = __ffs(parts) - 1;
                    parts &= parts - 1;
                    const Range range = buffers.tiles[size_t(tile) * MAX_BIN_PARTS + part];
                    ref = range.begin;
                    left = range.count;
                }
                const unsigned chunk = min(WARP_SIZE - filled, left);
                if (lane - filled < chunk) triId = buffers.tileRefs[ref + lane - filled];
                ref += chunk;
                left -= chunk;
                filled += chunk;
            } while (filled < WARP_SIZE && (parts || left));

            FineTriangle triangle{};
            if (lane < filled) {
                const auto tri = buffers.triangles[triId];
                triangle.coverage = triangleMask(tri, x0, y0) & valid;
                if (triangle.coverage) setupFragmentPlane(triangle, tri, x0, y0);
            }
            shadeBatch(triangle, lane, warpBatch[warp], pixels);
        }

        #pragma unroll
        for (unsigned slot = lane; slot < 64; slot += WARP_SIZE) {
            const int x = x0 + int(slot & 7), y = y0 + int(slot >> 3);
            if ((valid >> slot) & 1) framebuffer[size_t(y) * grid.width + x] = pixels[slot];
        }
        __syncwarp();
    }
}

// Never drop geometry on scratch overflow. Fine does not write a partial frame: the
// same stream runs the original rasterizer instead. The no-overflow launch is O(SMs).
__global__ void rasterizeOverflow(const float *vertices, const unsigned *faces, uint64_t *framebuffer,
                                  ProjectionGPU projection, unsigned nfaces, RasterGrid grid,
                                  const RasterCounters *counters)
{
    if (!counters->overflow) return;
    const unsigned stride = blockDim.x * gridDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < nfaces; i += stride) {
        rasterizeFace(vertices, faces, framebuffer, projection, i, grid.width, grid.height);
    }
}

template<class T>
class ScratchArray {
public:
    ScratchArray() = default;
    ~ScratchArray() { if (ptr) cudaFree(ptr); }
    ScratchArray(const ScratchArray &) = delete;
    ScratchArray &operator=(const ScratchArray &) = delete;

    void allocate(size_t count) { CUDA_SAFE_CALL(cudaMalloc(&ptr, count * sizeof(T))); }
    T *ptr = nullptr;
};
