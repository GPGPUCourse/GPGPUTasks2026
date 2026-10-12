#pragma once

#include <libgpu/shared_device_buffer.h>

namespace cuda {

struct StrassenPrepared {
    gpu::gpu_mem_16u a;
    gpu::gpu_mem_16u b;
    unsigned int w = 0;
    unsigned int h = 0;
    unsigned int k = 0;
    int device = -1;
    bool ready = false;
};

unsigned int matrix_strassen_variant_count();
const char* matrix_strassen_variant_name(unsigned int variant);
void matrix_strassen_describe(unsigned int variant);
void matrix_strassen_prepare(const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
    StrassenPrepared& prepared, unsigned int w, unsigned int h, unsigned int k);
void matrix_strassen_run(const StrassenPrepared& prepared, gpu::gpu_mem_32f& c,
    unsigned int variant);

}
