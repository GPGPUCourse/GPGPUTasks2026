#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Группа 32x8: варп — 32 соседних пикселя одной строки.
// Поток считает две соседние строки сразу, чтобы цепочки x,y не ждали друг друга.
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void mandelbrot(__global float* results,
                     unsigned int width, unsigned int height,
                     float fromX, float fromY,
                     float sizeX, float sizeY,
                     unsigned int iters, unsigned int isSmoothing)
{
    const uint i = get_global_id(0);
    const uint j0 = get_global_id(1) << 1;
    if (i >= width || j0 >= height)
        return;

    const uint j1 = j0 + 1u;
    const int haveB = j1 < height;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const float x0 = fromX + (i + 0.5f) * sizeX / (float)width;
    const float yA0 = fromY + (j0 + 0.5f) * sizeY / (float)height;
    const float yB0 = fromY + (j1 + 0.5f) * sizeY / (float)height;

    float xA = x0;
    float yA = yA0;
    float xB = x0;
    float yB = yB0;
    int iterA = 0;
    int iterB = 0;
    int deadA = 0;
    int deadB = haveB ? 0 : 1;

    #pragma unroll 1
    for (uint k = 0u; k < iters; ++k) {
        if (deadA & deadB)
            break;

        const float xPrevA = xA;
        const float xPrevB = xB;
        const float xnA = xPrevA * xPrevA - yA * yA + x0;
        const float ynA = 2.0f * xPrevA * yA + yA0;
        const float xnB = xPrevB * xPrevB - yB * yB + x0;
        const float ynB = 2.0f * xPrevB * yB + yB0;
        const int escA = (xnA * xnA + ynA * ynA) > threshold2;
        const int escB = (xnB * xnB + ynB * ynB) > threshold2;

        if (!deadA) {
            xA = xnA;
            yA = ynA;
            if (escA)
                deadA = 1;
            else
                ++iterA;
        }
        if (!deadB) {
            xB = xnB;
            yB = ynB;
            if (escB)
                deadB = 1;
            else
                ++iterB;
        }
    }

    float resultA = (float)iterA;
    float resultB = (float)iterB;
    if (isSmoothing) {
        if (iterA != iters) {
            resultA = resultA - log(log(sqrt(xA * xA + yA * yA)) / log(threshold)) / log(2.0f);
        }
        if (haveB && iterB != iters) {
            resultB = resultB - log(log(sqrt(xB * xB + yB * yB)) / log(threshold)) / log(2.0f);
        }
    }
    resultA = resultA / (float)iters;
    resultB = resultB / (float)iters;

    results[j0 * width + i] = resultA;
    if (haveB)
        results[j1 * width + i] = resultB;
}
