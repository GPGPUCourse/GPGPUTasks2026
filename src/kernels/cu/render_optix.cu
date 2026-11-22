#include "optix_params.cuh"
#include <optix_device.h>
#include "../defines.h"
#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"
#include "ambient_occlusion.cuh"

extern "C" {
__constant__ OptixRenderParams params;
}

extern "C" __global__ void __closesthit__triangle()
{
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __anyhit__shadow()
{
    if (optixGetPrimitiveIndex() == optixGetPayload_0())
        optixIgnoreIntersection();
    optixSetPayload_1(1);
    optixTerminateRay();
}

extern "C" __global__ void __raygen__render_optix()
{
    uint3 pixel = optixGetLaunchIndex();
    unsigned idx = pixel.y * params.camera.K.width + pixel.x;
    float3 origin, direction;
    make_primary_ray(params.camera, pixel.x + .5f, pixel.y + .5f, origin, direction);
    unsigned face = ~0u, distance = __float_as_uint(FLT_MAX);
    optixTrace(params.scene, origin, direction, 1e-6f, FLT_MAX, 0.f, 255,
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, face, distance);
    params.depth[idx] = __uint_as_float(distance);
    params.faceIds[idx] = (int)face;
    params.ao[idx] = ambient_occlusion(params.vertices, params.faces, origin, direction,
        (int)face, __uint_as_float(distance), idx, [](float3 p, float3 d, int ignoreFace) {
            unsigned ignored = ignoreFace, hit = 0;
            // Shadow ray using the terminate-on-first-hit flag. https://github.com/ingowald/optix7course/blob/6fafedcddac6df2ee367ed7eb08c8a982d1db069/example09_shadowRays/devicePrograms.cu#L135-L158
            optixTrace(params.scene, p, d, 1e-4f, FLT_MAX, 0.f, 255,
                       OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, ignored, hit);
            return hit != 0;
        });
}
