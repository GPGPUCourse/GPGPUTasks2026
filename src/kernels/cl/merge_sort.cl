#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#define UMAX (~0)

int lb(__local uint* arr, uint x, int n) {
	int l = -1, r = n;
	while (r - l != 1) {
		int m = (l + r) >> 1;
        if (arr[m] < x) l = m;
        else r = m;
	}
	return l + 1;
}

int rb(__local uint* arr, uint x, int n) {
	int l = -1, r = n;
	while (r - l != 1) {
		int m = (l + r) >> 1;
        if (arr[m] <= x) l = m;
        else r = m;
	}
	return l + 1;
}

int glb(__global const uint* arr, uint x, int n) {
	int l = -1, r = n;
	while (r - l != 1) {
		int m = (l + r) >> 1;
        if (arr[m] < x) l = m;
        else r = m;
	}
	return l + 1;
}

int grb(__global const uint* arr, uint x, int n) {
	int l = -1, r = n;
	while (r - l != 1) {
		int m = (l + r) >> 1;
        if (arr[m] <= x) l = m;
        else r = m;
	}
	return l + 1;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   uint  n)
{
    const unsigned int I = get_global_id(0);
    const unsigned int j = get_local_id(0);
    __local uint part[256];
    __local uint part_tmp[256];
    if (sorted_k == 0) {
        part[j] = (I < n) ? input_data[I] : UMAX;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < 8; ++i) {
            __local uint* in = !(i & 1) ? part : part_tmp;
            __local uint* out = (i & 1) ? part : part_tmp;
            int k = (1 << i);
            uint pos = (j & (k - 1));
            uint st = j - (j & ((k << 1) - 1));
            if (j & k) out[rb(in + st, in[j], k) + pos + st] = in[j];
            else out[lb(in + st + k, in[j], k) + pos + st] = in[j];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (I < n) {
            output_data[I] = part[j];
        }
    } else if (I < n) {
        const uint K = (1u << (sorted_k - 1)) * GROUP_SIZE;
        const uint x = input_data[I];
        const uint pos = (I & (K - 1));
        const uint st = I - (I & ((K << 1) - 1));
        const uint left_len = min(K, n - st);
        const uint right_len = (n - st > K) ? min(K, n - st - K) : 0;
        if (I & K) {
            output_data[grb(input_data + st, x, left_len) + pos + st] = x;
        } else {
            output_data[glb(input_data + st + K, x, right_len) + pos + st] = x;
        }
    }
}
