#ifndef MERGE_GLSL
#define MERGE_GLSL

#include "../defines.h"

shared uint merge_src[GROUP_SIZE];
shared uint merge_dst[GROUP_SIZE];

uint lower_bound(uint x, int left, int right, bool inv) {
    --left;
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (inv && merge_dst[mid] < x || !inv && merge_src[mid] < x) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return right;
}

uint upper_bound(uint x, int left, int right, bool inv) {
    --left;
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (inv && merge_dst[mid] <= x || !inv && merge_src[mid] <= x) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return right;
}

// [left, mid) & [mid, right).
void merge(uint i, uint left, uint mid, uint right, bool inverted)
{
    const uint val = inverted ? merge_dst[i] : merge_src[i];
    const uint rest = i < mid ? lower_bound(val, int(mid), int(right), inverted) - mid : upper_bound(val, int(left), int(mid), inverted) - left;
    const uint offset = i < mid ? i - left : i - mid;

    if (inverted) {
        merge_src[left + offset + rest] = val;
    } else {
        merge_dst[left + offset + rest] = val;
    }
}

#endif