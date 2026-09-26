#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    const unsigned int id = get_global_id(0);

    if (id >= width * height) {
        return;
    }

    const unsigned int x = id % height;
    const unsigned int y = id / height;

    const unsigned int index = x * width + y;
    c[index] = a[index] + b[index];
}
