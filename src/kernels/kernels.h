#pragma once

#include <libgpu/vulkan/engine.h>

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);
void fill_buffer_with_zeros(
        const gpu::WorkSize &workSize, 
           gpu::gpu_mem_32u &buffer, 
               unsigned int  n, 
               unsigned int  zero
    );
void radix_first_reduction(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &inputNums, 
           gpu::gpu_mem_32u &outBuf, 
               unsigned int  n,
               unsigned int  digitNum,
               unsigned int  reductionCoef
    );
void radix_generic_reduction(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &inputBuf,
                     size_t  inputOffset,
           gpu::gpu_mem_32u  &outBuf,
                     size_t  outOffset,
               unsigned int  n,
               unsigned int  reductionCoef
    );
void radix_generic_accumulation(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &sumBuf,
               unsigned int  sumBufOffset,
     const gpu::gpu_mem_32u &numBuf,
               unsigned int  numBufOffset,
     const gpu::gpu_mem_32u &outBuf,
               unsigned int  outBufOffset,
               unsigned int  n,
               unsigned int  reductionCoef
    );
void radix_last_accumulation(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &sumBuf,
     const gpu::gpu_mem_32u &numBuf,
     const gpu::gpu_mem_32u &outBuf,
               unsigned int  n,
               unsigned int digitNum,
               unsigned int  reductionCoef
    );
void radix_scatter(
        const gpu::WorkSize &workSize,
     const gpu::gpu_mem_32u &inputNums,
     const gpu::gpu_mem_32u &outNums,
     const gpu::gpu_mem_32u &prefixSums,
         const unsigned int  n,
         const unsigned int  digitNum

    );
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getFillBufferWithZeros();
const ProgramBinaries& getRadixSort01LocalCounting();
const ProgramBinaries& getRadixSort02GlobalPrefixesScanSumReduction();
const ProgramBinaries& getRadixSort03GlobalPrefixesScanAccumulation();
const ProgramBinaries& getRadixSort04Scatter();
}
