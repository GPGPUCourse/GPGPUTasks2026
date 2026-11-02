// Normalize float3
static __device__ float3 normalize3(const float3 v) {
    const float inv = rsqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return make_float3(v.x*inv, v.y*inv, v.z*inv);
}

// Make primary ray through pixel coordinates (u, v), f.e. (u, v) can be equal to (i+0.5, j+0.5)
static __device__ void make_primary_ray(const CameraViewGPU& cam, float u, float v, float3& ray_o, float3& ray_d)
{
    // 1) (u, v) - pixel center

    // 2) pinhole in camera space
    const float x_cam = (u - cam.K.cx) / cam.K.fx;
    const float y_cam = -(v - cam.K.cy) / cam.K.fy; // flip image Y -> camera Y
    const float z_cam = -1.0f;                      // look along -Z

    // 3) dir_world = R^T * dir_cam  (R is row-major 3x3)
    const float* R = cam.E.R;
    const float dx = R[0]*x_cam + R[3]*y_cam + R[6]*z_cam;
    const float dy = R[1]*x_cam + R[4]*y_cam + R[7]*z_cam;
    const float dz = R[2]*x_cam + R[5]*y_cam + R[8]*z_cam;

    // 4) origin = camera center in world; direction normalized
    ray_o = make_float3(cam.E.C[0], cam.E.C[1], cam.E.C[2]);
    ray_d = normalize3(make_float3(dx, dy, dz));
}