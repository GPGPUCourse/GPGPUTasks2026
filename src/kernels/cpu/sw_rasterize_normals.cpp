#include "../kernels.h"
#include "../../io/scene_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// 1 — OpenMP + atomic CAS; 0 — один поток, без атомиков.
#ifndef CPU_RASTERIZATION_MULTITHREADED
#define CPU_RASTERIZATION_MULTITHREADED 1
#endif

#if CPU_RASTERIZATION_MULTITHREADED
#include <atomic>
#endif

namespace cpu {
namespace {

using Visibility = unsigned long long;
constexpr Visibility NO_FRAGMENT = ~Visibility{0};
#if CPU_RASTERIZATION_MULTITHREADED
using Pixel = std::atomic<Visibility>;
#else
using Pixel = Visibility;
#endif
constexpr int SUBPIXEL = 256;
constexpr int HALF_PIXEL = SUBPIXEL / 2;

struct ScreenVertex {
    int x, y; // субпиксельные координаты, Y направлена вниз
    float z;  // глубина после деления на w
};

float projectRow(const float *row, point3f p)
{
    return fmaf(row[0], p.x, fmaf(row[1], p.y, fmaf(row[2], p.z, row[3])));
}

point4f project(const ProjectionGPU &projection, point3f p)
{
    const auto &r = projection.rows;
    return {projectRow(r[0], p), projectRow(r[1], p), projectRow(r[2], p), projectRow(r[3], p)};
}

float clamp01(float x) { return x > 0.0f ? fminf(1.0f, x) : 0.0f; }

uint32_t encodeNormal(float n)
{
    return static_cast<uint32_t>(roundf(255.0f * clamp01(0.5f * n + 0.5f)));
}

uint32_t shadeFragment(point3f a, point3f b, point3f c)
{
    const point3f u = b - a, v = c - a;
    const point3f n{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
    const float length = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (!(length > 0.0f)) return 0;
    return encodeNormal(n.x / length) | (encodeNormal(n.y / length) << 8) | (encodeNormal(n.z / length) << 16);
}

Visibility visibilityKey(float depth, uint32_t color)
{
    // Биты неотрицательных float сортируются как целые. При равной глубине сравниваем RGB.
    uint32_t depth_bits;
    std::memcpy(&depth_bits, &depth, sizeof(depth_bits));
    return (Visibility{depth_bits} << 32) | color;
}

ScreenVertex toScreen(point4f p, int width, int height)
{
    return {static_cast<int>(roundf(clamp01(0.5f * (p.x / p.w + 1.0f)) * width * SUBPIXEL)),
            static_cast<int>(roundf(clamp01(0.5f * (p.y / p.w + 1.0f)) * height * SUBPIXEL)),
            clamp01(p.z / p.w)};
}

int firstRow(int y)
{
    // Первая строка, центр пикселя которой не выше y.
    return (y - HALF_PIXEL + SUBPIXEL - 1) / SUBPIXEL;
}

struct Span {
    int left, right;
};

// Обходим ребро сверху вниз. На каждой строке находим первый центр пикселя
// на ребре или справа от него. Брезенхем: двигаем x целым шагом, а остаток
// накапливаем в error и при необходимости добавляем ещё один пиксель.
void walkEdge(ScreenVertex a, ScreenVertex b, std::vector<Span> &spans)
{
    const bool is_left = a.y > b.y;
    if (is_left) std::swap(a, b);
    const int first = firstRow(a.y), end = firstRow(b.y);
    if (first == end) return;

    const int64_t dx = b.x - a.x, dy = b.y - a.y;
    const int64_t divisor = dy * SUBPIXEL;
    const int64_t crossing = (a.x - HALF_PIXEL) * dy + dx * (first * SUBPIXEL + HALF_PIXEL - a.y);
    int x = int(crossing / divisor + (crossing % divisor > 0)); // ceil
    int64_t error = x * divisor - crossing;
    int step = int(dx / dy);
    int64_t remainder = (dx % dy) * SUBPIXEL;
    if (remainder < 0) {
        --step;
        remainder += divisor;
    }

    for (int y = first; y < end; ++y) {
        if (is_left) spans[y].left = x;
        else spans[y].right = x;
        x += step;
        error -= remainder;
        if (error < 0) {
            ++x;
            error += divisor;
        }
    }
}

void drawTriangle(ScreenVertex a, ScreenVertex b, ScreenVertex c, uint32_t color,
                  std::vector<Span> &spans, std::vector<Pixel> &framebuffer, int width)
{
    int64_t area = int64_t(b.x - a.x) * (c.y - a.y) - int64_t(b.y - a.y) * (c.x - a.x);
    if (area == 0) return;
    if (area < 0) {
        std::swap(b, c);
        area = -area;
    }
    walkEdge(a, b, spans);
    walkEdge(b, c, spans);
    walkEdge(c, a, spans);

    // Глубина линейна в экранных координатах.
    const float dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / float(area);
    const float dzdy = ((b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z)) / float(area);
    const int first = firstRow(std::min({a.y, b.y, c.y}));
    const int end = firstRow(std::max({a.y, b.y, c.y}));
    for (int y = first; y < end; ++y) {
        const int left = std::max(0, spans[y].left);
        const int right = std::min(width, spans[y].right);
        const float row_depth = fmaf(dzdy, float(y * SUBPIXEL + HALF_PIXEL - a.y), a.z);
        // Правый край не включаем, чтобы соседние треугольники не рисовали его дважды.
        for (int x = left; x < right; ++x) {
            const float depth = fmaf(dzdx, float(x * SUBPIXEL + HALF_PIXEL - a.x), row_depth);
            const Visibility fragment = visibilityKey(clamp01(depth), color);
            auto &pixel = framebuffer[size_t(y) * width + x];
#if CPU_RASTERIZATION_MULTITHREADED
            // Глубина и цвет обновляются одним CAS. При конфликте CAS вернёт новый ключ в previous.
            Visibility previous = pixel.load(std::memory_order_relaxed);
            while (fragment < previous && !pixel.compare_exchange_weak(previous, fragment, std::memory_order_relaxed)) {
            }
#else
            if (fragment < pixel) pixel = fragment;
#endif
        }
    }
}

float planeDistance(point4f p, int plane)
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

point4f clipIntersection(point4f a, point4f b, float da, float db)
{
    // Интерполируем от внутренней точки, чтобы общие рёбра отсекались одинаково.
    if (da < 0.0f) {
        std::swap(a, b);
        std::swap(da, db);
    }
    const float t = da / (da - db);
    return {fmaf(t, b.x - a.x, a.x), fmaf(t, b.y - a.y, a.y),
            fmaf(t, b.z - a.z, a.z), fmaf(t, b.w - a.w, a.w)};
}

void rasterizeTriangle(point4f a, point4f b, point4f c, uint32_t color,
                       std::vector<Span> &spans, std::vector<Pixel> &framebuffer, int width, int height)
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
        drawTriangle(toScreen(a, width, height), toScreen(b, width, height), toScreen(c, width, height),
                     color, spans, framebuffer, width);
        return;
    }

    // Отсечение Сазерленда — Ходжмана: каждая плоскость добавляет не более одной вершины.
    point4f polygons[2][9] = {{a, b, c}};
    int input = 0, size = 3;
    for (int plane = 0; plane < 6; ++plane) {
        if (!(clip_mask & (1u << plane))) continue;
        int count = 0;
        point4f prev = polygons[input][size - 1];
        float dp = planeDistance(prev, plane);
        for (int i = 0; i < size; ++i) {
            const point4f curr = polygons[input][i];
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
        drawTriangle(first, toScreen(polygon[i - 1], width, height), toScreen(polygon[i], width, height),
                     color, spans, framebuffer, width);
    }
}

} // namespace

image8u sw_rasterize_normals(const SceneGeometry &scene, const ProjectionGPU &projection, int width, int height)
{
#if CPU_RASTERIZATION_MULTITHREADED
    std::vector<Pixel> framebuffer(size_t(width) * height);
    for (auto &pixel : framebuffer) pixel.store(NO_FRAGMENT, std::memory_order_relaxed);
    #pragma omp parallel
#else
    std::vector<Pixel> framebuffer(size_t(width) * height, NO_FRAGMENT);
#endif
    {
        std::vector<Span> spans(height); // Отдельные границы строк у каждого потока.
#if CPU_RASTERIZATION_MULTITHREADED
        #pragma omp for schedule(guided)
#endif
        for (size_t i = 0; i < scene.faces.size(); ++i) {
            const point3u &face = scene.faces[i];
            const point3f &v0 = scene.vertices[face.x];
            const point3f &v1 = scene.vertices[face.y];
            const point3f &v2 = scene.vertices[face.z];
            const uint32_t color = shadeFragment(v0, v1, v2);
            rasterizeTriangle(project(projection, v0), project(projection, v1), project(projection, v2),
                              color, spans, framebuffer, width, height);
        }
    }

    // Все потоки закончили: распаковываем RGB победителей, без повторного шейдинга.
    image8u image(width, height, 3);
    for (int y = 0; y < height; ++y) {
        auto *row = image.ptr(y);
        for (int x = 0; x < width; ++x) {
            const auto &pixel = framebuffer[size_t(y) * width + x];
#if CPU_RASTERIZATION_MULTITHREADED
            const Visibility fragment = pixel.load(std::memory_order_relaxed);
#else
            const Visibility fragment = pixel;
#endif
            const uint32_t color = fragment == NO_FRAGMENT ? 0u : static_cast<uint32_t>(fragment);
            row[3 * x + 0] = color & 255u;
            row[3 * x + 1] = (color >> 8) & 255u;
            row[3 * x + 2] = (color >> 16) & 255u;
        }
    }
    return image;
}

} // namespace cpu
