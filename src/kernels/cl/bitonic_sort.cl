#include "../defines.h"
#include "helpers/rassert.cl"

// Gather each tile; let bitonic waves align,
// UINT_MAX fills the void, and ordered threads shine.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
bitonic_sort(
    __global const uint* input_data,
    __global uint* output_data,
    uint n)
{
    const uint weft = get_local_id(0);
    const uint place = get_global_id(0);

    __local uint loom[GROUP_SIZE];

    loom[weft] = place < n ? input_data[place] : UINT_MAX;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint wave = 2u; wave <= GROUP_SIZE; wave <<= 1u) {
        for (uint stride = wave >> 1u; stride; stride >>= 1u) {
            const uint mate = weft ^ stride;

            if (mate > weft) {
                const uint here = loom[weft];
                const uint there = loom[mate];

                if ((weft & wave) ? here < there : here > there) {
                    loom[weft] = there;
                    loom[mate] = here;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (place < n)
        output_data[place] = loom[weft];
}