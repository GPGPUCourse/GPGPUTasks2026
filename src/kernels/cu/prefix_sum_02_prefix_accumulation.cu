#include <libgpu/cuda/utils.h>
#include "../kernels.h"
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>


namespace {

__global__ void prepare_first_element_kernel(
    unsigned int* __restrict__ input_ptr,
    unsigned int orig_first,
    const unsigned int* s0,
    const unsigned int* s1,
    const unsigned int* s2,
    int dev)
{
    unsigned int val = orig_first + *s0;
    if (dev >= 2) val += *s1;
    if (dev >= 3) val += *s2;
    *input_ptr = val;
}

__global__ void restore_first_element_kernel(
    unsigned int* __restrict__ input_ptr,
    unsigned int orig_first)
{
    *input_ptr = orig_first;
}

struct DeviceState {
    cudaStream_t stream = nullptr;
    cudaEvent_t reduce_event = nullptr;
    void* d_reduce_temp = nullptr;
    size_t reduce_temp_bytes = 0;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    unsigned int* d_chunk_sum = nullptr;
    unsigned int orig_first_elem = 0;
};

static ::std::vector<DeviceState> g_devs;

void ensure_initialized(int num_devices, const ::std::vector<unsigned int>& counts,
                        const ::std::vector<unsigned int*>& inputs, const ::std::vector<unsigned int*>& outputs)
{
    if (!g_devs.empty()) {
        return;
    }
    g_devs.resize(num_devices);

    for (int i = 0; i < num_devices; ++i) {
        CUDA_SAFE_CALL(cudaSetDevice(i));
        for (int j = 0; j < num_devices; ++j) {
            if (i != j) {
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_SAFE_CALL(err);
                }
            }
        }
        CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&g_devs[i].stream, cudaStreamNonBlocking));

        if (num_devices > 1) {
            CUDA_SAFE_CALL(cudaMemcpy(&g_devs[i].orig_first_elem, inputs[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        if (num_devices > 1 && i + 1 < num_devices) {
            CUDA_SAFE_CALL(cudaEventCreateWithFlags(&g_devs[i].reduce_event, cudaEventDisableTiming));
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&g_devs[i].d_chunk_sum), sizeof(unsigned int)));
            CUDA_SAFE_CALL(cub::DeviceReduce::Sum(
                nullptr, g_devs[i].reduce_temp_bytes,
                inputs[i], g_devs[i].d_chunk_sum, counts[i], g_devs[i].stream));
            CUDA_SAFE_CALL(cudaMalloc(&g_devs[i].d_reduce_temp, g_devs[i].reduce_temp_bytes));
        }

        CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
            nullptr, g_devs[i].scan_temp_bytes,
            inputs[i], outputs[i], counts[i], g_devs[i].stream));
        CUDA_SAFE_CALL(cudaMalloc(&g_devs[i].d_scan_temp, g_devs[i].scan_temp_bytes));
    }
}

} // anonymous namespace

namespace cuda {

void prefix_sum_02_prefix_accumulation(int num_devices, const ::std::vector<unsigned int>& counts,
                                       const ::std::vector<unsigned int*>& inputs, const ::std::vector<unsigned int*>& outputs)
{
    ensure_initialized(num_devices, counts, inputs, outputs);

    if (num_devices == 1) {
        CUDA_SAFE_CALL(cudaSetDevice(0));
        CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
            g_devs[0].d_scan_temp, g_devs[0].scan_temp_bytes,
            inputs[0], outputs[0], counts[0], g_devs[0].stream));
        return;
    }

    // Phase 1
    for (int dev = 0; dev + 1 < num_devices; ++dev) {
        CUDA_SAFE_CALL(cudaSetDevice(dev));
        auto& d = g_devs[dev];
        CUDA_SAFE_CALL(cub::DeviceReduce::Sum(
            d.d_reduce_temp, d.reduce_temp_bytes,
            inputs[dev], d.d_chunk_sum, counts[dev], d.stream));
        CUDA_SAFE_CALL(cudaEventRecord(d.reduce_event, d.stream));
        if (dev == 0) {
            CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
                d.d_scan_temp, d.scan_temp_bytes,
                inputs[0], outputs[0], counts[0], d.stream));
        }
    }

    // Phase 2
    for (int dev = 1; dev < num_devices; ++dev) {
        CUDA_SAFE_CALL(cudaSetDevice(dev));
        auto& d = g_devs[dev];
        for (int p = 0; p < dev; ++p) {
            CUDA_SAFE_CALL(cudaStreamWaitEvent(d.stream, g_devs[p].reduce_event, 0));
        }
        const unsigned int* s0 = g_devs[0].d_chunk_sum;
        const unsigned int* s1 = (dev >= 2) ? g_devs[1].d_chunk_sum : nullptr;
        const unsigned int* s2 = (dev >= 3) ? g_devs[2].d_chunk_sum : nullptr;
        prepare_first_element_kernel<<<1, 1, 0, d.stream>>>(
            inputs[dev], d.orig_first_elem, s0, s1, s2, dev);
        CUDA_CHECK_KERNEL_ASYNC(d.stream);

        CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
            d.d_scan_temp, d.scan_temp_bytes,
            inputs[dev], outputs[dev], counts[dev], d.stream));

        if (dev + 1 < num_devices) {
            restore_first_element_kernel<<<1, 1, 0, d.stream>>>(
                inputs[dev], d.orig_first_elem);
            CUDA_CHECK_KERNEL_ASYNC(d.stream);
        }
    }
}

} // namespace cuda
