#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../shared_structs/camera_gpu_shared.h"

// Normalize float3
static inline float3 normalize3(const float3 v)
{
    const float inv = rsqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return (float3)(v.x * inv, v.y * inv, v.z * inv);
}

// Make primary ray through pixel coordinates (u, v), e.g. (u, v) can be (i+0.5, j+0.5)
static inline void make_primary_ray(__global const CameraViewGPU* cam,
                                    float                         u,
                                    float                         v,
                                    __private float3*             ray_o,
                                    __private float3*             ray_d)
{
    // 1) (u, v) - pixel center

    // 2) pinhole in camera space
    const float x_cam = (u - cam->K.cx) / cam->K.fx;
    const float y_cam = -(v - cam->K.cy) / cam->K.fy; // flip image Y -> camera Y
    const float z_cam = -1.0f;                        // look along -Z

    // 3) dir_world = R^T * dir_cam  (R is row-major 3x3)
    const float dx = cam->E.R[0] * x_cam + cam->E.R[3] * y_cam + cam->E.R[6] * z_cam;
    const float dy = cam->E.R[1] * x_cam + cam->E.R[4] * y_cam + cam->E.R[7] * z_cam;
    const float dz = cam->E.R[2] * x_cam + cam->E.R[5] * y_cam + cam->E.R[8] * z_cam;

    // 4) origin = camera center in world; direction normalized
    *ray_o = (float3)(cam->E.C[0], cam->E.C[1], cam->E.C[2]);
    *ray_d = normalize3((float3)(dx, dy, dz));
}
