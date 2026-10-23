#pragma once
#include "defines.h"

#include <libgpu/vulkan/engine.h>
#define NUM_BANKS 32

__device__ __forceinline__ unsigned int compute_index(unsigned int n)
{
    return n + (n / NUM_BANKS);
}

__device__ __forceinline__ void blelloch_local_scan(unsigned int* temp, unsigned int* partial_sums, unsigned int data_size)
{
    unsigned int offset = 1;

#pragma unroll
    for (unsigned int d = data_size >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (threadIdx.x < d) {
            unsigned int ai = offset * (2 * threadIdx.x + 1) - 1;
            unsigned int bi = ai + offset;
            ai = compute_index(ai);
            bi = compute_index(bi);
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        if (partial_sums != nullptr) {
            partial_sums[blockIdx.x] = temp[compute_index(data_size - 1)];
        }
        temp[compute_index(data_size - 1)] = 0;
    }

#pragma unroll
    for (unsigned int d = 1; d < data_size; d <<= 1) {
        __syncthreads();
        offset >>= 1;

        if (threadIdx.x < d) {

            unsigned int first = (2 * threadIdx.x + 1) * offset - 1;
            unsigned int second = first + offset;
            first = compute_index(first);
            second = compute_index(second);

            unsigned int first_buf = temp[first];
            temp[first] = temp[second];
            temp[second] += first_buf;
        }
    }

    __syncthreads();
}