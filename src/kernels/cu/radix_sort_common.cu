#pragma once

#include "../defines.h"

static __device__ void workgroupExclusiveScan(unsigned int *data, unsigned int *scratch, unsigned int ipt)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int wg = blockDim.x;
    const unsigned int from = lid * ipt;

    unsigned int sum = 0;
    for (unsigned int k = 0; k < ipt; ++k) {
        sum += data[from + k];
    }

    scratch[lid] = sum;
    __syncthreads();

    for (unsigned int d = 1; d < wg; d <<= 1) {
        const unsigned int add = (lid >= d) ? scratch[lid - d] : 0;
        __syncthreads();
        scratch[lid] += add;
        __syncthreads();
    }

    unsigned int running = scratch[lid] - sum;
    for (unsigned int k = 0; k < ipt; ++k) {
        const unsigned int value = data[from + k];
        data[from + k] = running;
        running += value;
    }
    __syncthreads();
}
