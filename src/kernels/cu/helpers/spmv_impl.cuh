#pragma once

namespace spmv7 {

__global__ void multiply(const unsigned int* __restrict__ row_offsets,
                        const unsigned int* __restrict__ columns,
                        const unsigned int* __restrict__ values,
                        const unsigned int* __restrict__ vector_values,
                        unsigned int* __restrict__ output,
                        unsigned int nrows)
{
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= nrows) {
        return;
    }

    const unsigned int begin = row_offsets[row];
    const unsigned int end = row_offsets[row + 1];
    unsigned int sum = 0;
    for (unsigned int i = begin + lane; i < end; i += 32) {
        sum += values[i] * vector_values[columns[i]];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

} // namespace spmv7
