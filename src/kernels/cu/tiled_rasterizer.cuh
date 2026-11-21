// Included by sw_rasterize_normals.cu: shares its projection, clipping and Bresenham rules.

constexpr int TILE_SIZE = 8;
constexpr int BIN_SIZE = 128;
constexpr int TILES_PER_BIN = BIN_SIZE / TILE_SIZE;
constexpr int BIN_THREADS = 256;
constexpr int COARSE_THREADS = TILES_PER_BIN * TILES_PER_BIN;
constexpr int FINE_THREADS = TILE_SIZE * TILE_SIZE;
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

__global__ void triangleSetup(const float *vertices, const unsigned *faces, ProjectionGPU projection,
                              unsigned nfaces, RasterGrid grid, RasterBuffers buffers)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfaces) return;
    const uint3 face = loadFace(faces, i);
    const float3 a = loadVertex(vertices, face.x), b = loadVertex(vertices, face.y), c = loadVertex(vertices, face.z);
    clipAndProject(project(projection, a), project(projection, b), project(projection, c), grid.width, grid.height,
                   EmitTriangle{buffers, a, b, c});
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

__device__ void drawInTile(const SetupTriangle &tri, int x0, int y0, int x1, int y1, PixelKey *tile)
{
    const auto a = tri.a, b = tri.b, c = tri.c;
    const auto bounds = pixelBounds(a, b, c);
    const int first = max(y0, bounds.y0), end = min(y1, bounds.y1);
    const float area = float(edgeValue(a, b, c.x, c.y));
    const float dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / area;
    const float dzdy = ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)) / area;
    // Setup already tested coverage when the entire bounding box contains exactly one sample.
    if (bounds.x1 == bounds.x0 + 1 && bounds.y1 == bounds.y0 + 1) {
        const int x = bounds.x0, y = bounds.y0;
        const float rowDepth = fmaf(dzdy, float(y * SUBPIXEL + HALF_PIXEL - a.y), a.z);
        const float depth = clamp01(fmaf(dzdx, float(x * SUBPIXEL + HALF_PIXEL - a.x), rowDepth));
        atomicMin(tile + (y - y0) * TILE_SIZE + x - x0, (PixelKey(__float_as_uint(depth)) << 32) | tri.color);
        return;
    }
    // Initialize each Bresenham edge directly at the first row owned by the tile.
    EdgeWalker edges[] = {{a, b, first, end, true}, {b, c, first, end, true}, {c, a, first, end, true}};
    for (int y = first; y < end; ++y) {
        int left = x0, right = x1;
        for (auto &edge : edges) {
            if (y < edge.first || y >= edge.end) continue;
            if (edge.left) left = max(x0, edge.ceilX());
            else right = min(x1, edge.ceilX());
            edge.next();
        }
        const float rowDepth = fmaf(dzdy, float(y * SUBPIXEL + HALF_PIXEL - a.y), a.z);
        for (int x = left; x < right; ++x) {
            const float depth = clamp01(fmaf(dzdx, float(x * SUBPIXEL + HALF_PIXEL - a.x), rowDepth));
            const PixelKey fragment = (PixelKey(__float_as_uint(depth)) << 32) | tri.color;
            atomicMin(tile + (y - y0) * TILE_SIZE + x - x0, fragment);
        }
    }
}

__global__ void fineRasterizer(RasterGrid grid, RasterBuffers buffers, uint64_t *framebuffer)
{
    if (buffers.counters->overflow) return;
    __shared__ PixelKey pixels[TILE_SIZE * TILE_SIZE];
    const int lane = threadIdx.x;
    for (unsigned i = blockIdx.x; i < buffers.counters->numActiveTiles; i += gridDim.x) {
        const int tile = buffers.activeTiles[i];
        const int x0 = (tile % grid.tilesX) * TILE_SIZE, y0 = (tile / grid.tilesX) * TILE_SIZE;
        for (int slot = lane; slot < TILE_SIZE * TILE_SIZE; slot += FINE_THREADS) {
            const int x = x0 + slot % TILE_SIZE, y = y0 + slot / TILE_SIZE;
            const bool valid = x < grid.width && y < grid.height;
            pixels[slot] = valid ? framebuffer[size_t(y) * grid.width + x] : ~PixelKey{0};
        }
        __syncthreads();
        unsigned offset = lane;
        for (unsigned mask = buffers.tileParts[tile]; mask; mask &= mask - 1) {
            const int part = __ffs(mask) - 1;
            const Range tris = buffers.tiles[size_t(tile) * MAX_BIN_PARTS + part];
            while (offset < tris.count) {
                const auto tri = buffers.triangles[buffers.tileRefs[tris.begin + offset]];
                drawInTile(tri, x0, y0, min(x0 + TILE_SIZE, grid.width), min(y0 + TILE_SIZE, grid.height), pixels);
                offset += FINE_THREADS;
            }
            offset -= tris.count;
        }
        __syncthreads();
        for (int slot = lane; slot < TILE_SIZE * TILE_SIZE; slot += FINE_THREADS) {
            const int x = x0 + slot % TILE_SIZE, y = y0 + slot / TILE_SIZE;
            if (x < grid.width && y < grid.height) framebuffer[size_t(y) * grid.width + x] = pixels[slot];
        }
        __syncthreads();
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
