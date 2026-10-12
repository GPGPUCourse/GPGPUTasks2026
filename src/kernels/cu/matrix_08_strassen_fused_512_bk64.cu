#include "helpers/strassen_fused.cuh"

namespace strassen_detail {

__global__ __launch_bounds__(512, 1) void strassen_fused_512_bk64(
    const half* __restrict__ a, const half* __restrict__ b, float* __restrict__ c,
    int hm, int wn, int hk)
{
    __shared__ __align__(16) half shared[4 * 64 * 64];
    fused<64, 512, false>(a, b, c, hm, wn, hk, shared);
}

}
