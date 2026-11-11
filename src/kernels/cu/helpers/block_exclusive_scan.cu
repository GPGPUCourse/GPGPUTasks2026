#pragma once

__device__ __forceinline__ unsigned int block_exclusive_scan(
    unsigned int* temp, unsigned int n_elems)
{
    const unsigned int thid = threadIdx.x;
    int offset = 1;

    // Up-Sweep
    for (int d = n_elems >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }


    __syncthreads();
    const unsigned int total = temp[n_elems - 1];
    __syncthreads();

    if (thid == 0) {
        temp[n_elems - 1] = 0;
    }

    // Down-Sweep
    for (int d = 1; d < n_elems; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    return total;
}
