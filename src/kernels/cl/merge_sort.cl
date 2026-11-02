#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// Бинарный поиск точки
int merge_path(__global const uint* A, int a_len,
    __global const uint* B, int b_len,
    int diag)
{
    int l = max(0, diag - b_len);
    int r = min(diag, a_len);

    while (r - l >= 1) {
        int mid = l + (r - l) / 2;
        int left = A[mid]; // "слева"
        int top = B[diag - 1 - mid]; // "сверху"
        if (left > top) {
            // слева больше чем сверху?
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    return l;
}

int get_len(int sorted_k, int n, int start)
{
    int len = sorted_k;
    if (start >= n) {
        len = 0;
    } else if (start + len > n) {
        len = n - start;
    }
    return len;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    //этот WorkItem отвечает за один элемент результата C
    int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    //   A = [pair*2*sorted_k, ...), B = A + sorted_k
    int pair = gid / (2 * sorted_k); // номер
    int k = gid - pair * (2 * sorted_k);

    int a_start = pair * 2 * sorted_k;
    int b_start = a_start + sorted_k;

    int a_len = get_len(sorted_k, n, a_start);
    int b_len = get_len(sorted_k, n, b_start);

    if (b_len == 0) {
        output_data[gid] = input_data[a_start + k];
        return;
    }

    __global const uint* A = input_data + a_start;
    __global const uint* B = input_data + b_start;

    int i = merge_path(A, a_len, B, b_len, k); // A
    int j = k - i; // B

    int c_val = 0;
    if (i >= a_len) {
        // todo: просмотреть все условия бдительно, мог накосячить
        c_val = B[j];
    } else if (j >= b_len) {
        c_val = A[i];
    } else if (A[i] <= B[j]) {
        c_val = A[i];
    } else {
        c_val = B[j];
    }
    output_data[gid] = c_val;
}
