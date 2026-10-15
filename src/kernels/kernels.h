#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(const gpu::WorkSize &workSize, gpu::gpu_mem_32u &buffer_fst, gpu::gpu_mem_32u &buffer_snd, unsigned int n);
void prefix_sum_01_reduction(
          const gpu::WorkSize  &workSize,
       const gpu::gpu_mem_32u  &data_buf, 
                    const int  data_offset,
       const gpu::gpu_mem_32u  &result_buf,
                    const int  result_offset,
                 unsigned int  n,
                 unsigned int  reduction_coef
);
void prefix_sum_02_prefix_accumulation(
                    const gpu::WorkSize  &workSize,
                 const gpu::gpu_mem_32u  &partial_sum_buf,
                           unsigned int  partial_sum_offset,
                 const gpu::gpu_mem_32u  &input_nums,
                           unsigned int  input_nums_offset,
                 const gpu::gpu_mem_32u  result_buf,
                           unsigned int  results_buf_offset,
                           unsigned int  n,
                           unsigned int  reduction_coef                                        
);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getPrefixSum01Reduction();
const ProgramBinaries& getPrefixSum02PrefixAccumulation();
}
