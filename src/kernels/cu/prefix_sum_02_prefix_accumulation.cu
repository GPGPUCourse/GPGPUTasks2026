#include <libgpu/cuda/utils.h>
#include "../kernels.h"
#include <cub/device/device_scan.cuh>

namespace {

__global__ void prefix_sum_accumulate_offset_kernel(
    unsigned int* __restrict__ data,
    unsigned int n,
    const unsigned int* const* prev_sums,
    unsigned int count)
{
    __shared__ unsigned int s_offset;
    if (threadIdx.x == 0) {
        unsigned int total = 0;
        for (unsigned int p = 0; p < count; ++p) {
            total += *prev_sums[p];
        }
        s_offset = total;
    }
    __syncthreads();

    unsigned int add_val = s_offset;
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        uint4 v = *reinterpret_cast<const uint4*>(data + idx);
        v.x += add_val;
        v.y += add_val;
        v.z += add_val;
        v.w += add_val;
        *reinterpret_cast<uint4*>(data + idx) = v;
    } else {
        for (unsigned int i = idx; i < n; ++i) {
            data[i] += add_val;
        }
    }
}

struct DeviceState {
    cudaStream_t stream = nullptr;
    cudaEvent_t scan_event = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    unsigned int* d_chunk_sum = nullptr;
    const unsigned int* h_prev_ptrs[4] = {nullptr};
    const unsigned int** d_prev_ptrs = nullptr;
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
        CUDA_SAFE_CALL(cudaEventCreateWithFlags(&g_devs[i].scan_event, cudaEventDisableTiming));

        CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
            nullptr, g_devs[i].temp_bytes,
            inputs[i], outputs[i], counts[i], g_devs[i].stream));
        CUDA_SAFE_CALL(cudaMalloc(&g_devs[i].d_temp, g_devs[i].temp_bytes));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&g_devs[i].d_chunk_sum), sizeof(unsigned int)));

        if (i > 0) {
            for (int p = 0; p < i; ++p) {
                g_devs[i].h_prev_ptrs[p] = g_devs[p].d_chunk_sum;
            }
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&g_devs[i].d_prev_ptrs), sizeof(unsigned int*) * i));
            CUDA_SAFE_CALL(cudaMemcpy(g_devs[i].d_prev_ptrs, g_devs[i].h_prev_ptrs, sizeof(unsigned int*) * i, cudaMemcpyHostToDevice));
        }
    }
}

} // anonymous namespace

namespace cuda {

void prefix_sum_02_prefix_accumulation(int num_devices, const ::std::vector<unsigned int>& counts,
                                       const ::std::vector<unsigned int*>& inputs, const ::std::vector<unsigned int*>& outputs)
{
    ensure_initialized(num_devices, counts, inputs, outputs);

    for (int dev = 0; dev < num_devices; ++dev) {
        CUDA_SAFE_CALL(cudaSetDevice(dev));
        auto& d = g_devs[dev];

        CUDA_SAFE_CALL(cub::DeviceScan::InclusiveSum(
            d.d_temp, d.temp_bytes,
            inputs[dev], outputs[dev], counts[dev], d.stream));

        if (num_devices > 1) {
            CUDA_SAFE_CALL(cudaMemcpyAsync(d.d_chunk_sum, outputs[dev] + counts[dev] - 1, sizeof(unsigned int),
                cudaMemcpyDeviceToDevice, d.stream));
            CUDA_SAFE_CALL(cudaEventRecord(d.scan_event, d.stream));

            if (dev > 0) {
                for (int p = 0; p < dev; ++p) {
                    CUDA_SAFE_CALL(cudaStreamWaitEvent(d.stream, g_devs[p].scan_event, 0));
                }
                unsigned int num_vec = (counts[dev] + 3) / 4;
                unsigned int grid_size = (num_vec + 255) / 256;
                prefix_sum_accumulate_offset_kernel<<<grid_size, 256, 0, d.stream>>>(
                    outputs[dev], counts[dev], d.d_prev_ptrs, dev);
                CUDA_CHECK_KERNEL_ASYNC(d.stream);
            }
        }
    }
}

} // namespace cuda
