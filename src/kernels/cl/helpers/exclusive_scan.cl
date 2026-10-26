inline uint exclusive_scan(__local uint* sums, uint lane)
{
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 1; stride < GROUP_SIZE; stride <<= 1) {
        uint index = (lane + 1) * (stride << 1) - 1;
        if (index < GROUP_SIZE)
            sums[index] += sums[index - stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    uint total = sums[GROUP_SIZE - 1];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) sums[GROUP_SIZE - 1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = GROUP_SIZE >> 1; stride > 0; stride >>= 1) {
        uint index = (lane + 1) * (stride << 1) - 1;
        if (index < GROUP_SIZE) {
            uint left = sums[index - stride];
            sums[index - stride] = sums[index];
            sums[index] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return total;
}
