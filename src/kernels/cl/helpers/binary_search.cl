#define BINARY_SEARCH(QUALIFIER) \
    uint binary_search(QUALIFIER uint* array, uint array_size, uint target, bool is_lower_bound) { \
        uint l = 0, r = array_size;                                                                \
        while (l < r) {                                                                            \
            uint m = l + (r - l) / 2;                                                              \
            uint value = array[m];                                                                 \
            if (is_lower_bound ? value < target : value <= target) {                               \
                l = m + 1;                                                                         \
            } else {                                                                               \
                r = m;                                                                             \
            }                                                                                      \
        }                                                                                          \
        return l;                                                                                  \
    }
