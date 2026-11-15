#pragma once

enum class chunk_side : bool {
    left,
    right
};

__device__ inline unsigned pos_bin_search(
    const unsigned* buf,
    unsigned num,
    unsigned size,
    chunk_side cs)
{
    auto g = [](unsigned a, unsigned b) { return a > b; };
    auto ge = [](unsigned a, unsigned b) { return a >= b; };
    auto cmp = (cs == chunk_side::right) ? g : ge;

    unsigned l = 0;
    unsigned r = size;
    unsigned p = (l + r) / 2;

    while (l != r) {
        if (cmp(buf[p], num))
            r = p;
        else
            l = p + 1;

        p = (l + r) / 2;
    }

    return p;
}
