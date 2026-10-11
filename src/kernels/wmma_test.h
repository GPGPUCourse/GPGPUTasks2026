#ifndef wmma_test
#define wmma_test

struct WmmaConfig {
    unsigned int blockH;
    unsigned int blockK;
    unsigned int blockW;
    unsigned int warpsH;
    unsigned int warpsW;
};

namespace cuda {
void matrix_multiply_wmma_test(
    const WmmaConfig& q,
    const gpu::gpu_mem_32f& a,
    const gpu::gpu_mem_32f& b,
    gpu::gpu_mem_32f& c,
    unsigned int w, unsigned int h, unsigned int k);
}

#endif // wmma_test