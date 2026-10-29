#include <libbase/stats.h>
#include <libutils/misc.h>
#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/device.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef CUDA_SUPPORT
#include <cuda_runtime_api.h>

// Объявления оставлены здесь, чтобы заменить только два файла, не трогая kernels.h.
namespace merge_fast {
int tile_size();
int partition_count(int n);
void merge_pass(const unsigned int* input, unsigned int* output,
                int* partitions, int run, int n, cudaStream_t stream);
void sort(const unsigned int* input, unsigned int* output,
          unsigned int* temporary, int* partitions, int n, cudaStream_t stream);
}

namespace {
void check_cuda(cudaError_t status, int line)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(cudaGetErrorString(status)) +
                                 " at line " + std::to_string(line));
    }
}
#define MS_CUDA(call) check_cuda((call), __LINE__)

int env_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    return value ? std::stoi(value) : fallback;
}

struct GpuBuffers {
    int device = 0;
    int offset = 0;
    int count = 0;
    unsigned int* input = nullptr;
    unsigned int* output = nullptr;
    unsigned int* temporary = nullptr;
    int* partitions = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
};

// Сначала завершаем работу ВСЕХ GPU, затем освобождаем память:
// на нее могут еще ссылаться P2P-копирования с другой карты.
struct Workspace {
    std::vector<GpuBuffers> gpus;
    ~Workspace()
    {
        for (auto& g : gpus) {
            if (g.stream) {
                cudaSetDevice(g.device);
                cudaStreamSynchronize(g.stream);
            }
        }
        for (auto& g : gpus) {
            cudaSetDevice(g.device);
            if (g.done) cudaEventDestroy(g.done);
            if (g.stream) cudaStreamDestroy(g.stream);
            if (g.partitions) cudaFree(g.partitions);
            if (g.temporary) cudaFree(g.temporary);
            if (g.output) cudaFree(g.output);
            if (g.input) cudaFree(g.input);
        }
    }
};

void allocate(GpuBuffers& g, int capacity)
{
    MS_CUDA(cudaSetDevice(g.device));
    MS_CUDA(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    MS_CUDA(cudaEventCreateWithFlags(&g.done, cudaEventDisableTiming));
    const size_t bytes = size_t(capacity) * sizeof(unsigned int);
    MS_CUDA(cudaMalloc(reinterpret_cast<void**>(&g.input), bytes));
    MS_CUDA(cudaMalloc(reinterpret_cast<void**>(&g.output), bytes));
    MS_CUDA(cudaMalloc(reinterpret_cast<void**>(&g.temporary), bytes));
    MS_CUDA(cudaMalloc(reinterpret_cast<void**>(&g.partitions),
                       size_t(merge_fast::partition_count(capacity)) * sizeof(int)));
}

void enable_peer(int from, int to)
{
    int possible = 0;
    MS_CUDA(cudaDeviceCanAccessPeer(&possible, from, to));
    if (!possible) {
        throw std::runtime_error("No P2P access between CUDA devices " +
                                 std::to_string(from) + " and " + std::to_string(to) +
                                 ". Run with MERGE_GPUS=1.");
    }
    MS_CUDA(cudaSetDevice(from));
    const cudaError_t status = cudaDeviceEnablePeerAccess(to, 0);
    if (status == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
    } else {
        MS_CUDA(status);
    }
}
} // namespace
#endif

void run(int argc, char** argv)
{
    const gpu::Device device = gpu::chooseGPUDevice(
        gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    if (!device.supports_cuda) {
        std::cout << "Device doesn't support CUDA; skipping CUDA experiment." << std::endl;
        return;
    }

#ifndef CUDA_SUPPORT
    throw std::runtime_error("Build this experiment with -DGPU_CUDA_SUPPORT=ON");
#else
    // Используем CUDA Runtime и primary contexts непосредственно, без activateContext().
    // Это позволяет явно переключать GPU, не смешивая их с контекстом libgpu.
    const int root_device = device.device_id_cuda;
    MS_CUDA(cudaSetDevice(root_device));
    std::cout << "Using CUDA API..." << std::endl;

    const int n = env_int("MERGE_N", 100 * 1000 * 1000);
    const int max_value = env_int("MERGE_MAX", std::numeric_limits<int>::max() - 1);
    const int iterations = env_int("MERGE_ITERS", 10);
    const int requested_gpus = env_int("MERGE_GPUS", 1);
    if (n < 1 || n > 1000000000 || max_value < 1 ||
        max_value > std::numeric_limits<int>::max() - 1 ||
        iterations < 1 || requested_gpus < 1 || requested_gpus > 4) {
        throw std::runtime_error("Use 1<=MERGE_N<=1e9, 1<=MERGE_MAX<=2147483646, "
                                 "MERGE_ITERS>=1, 1<=MERGE_GPUS<=4.");
    }

    int available = 0;
    MS_CUDA(cudaGetDeviceCount(&available));
    if (requested_gpus > available) {
        throw std::runtime_error("Requested more CUDA GPUs than are visible.");
    }
    std::vector<int> device_ids{root_device};
    for (int d = 0; d < available && int(device_ids.size()) < requested_gpus; ++d) {
        if (d != root_device) device_ids.push_back(d);
    }

    const int tile = merge_fast::tile_size();
    // Все межкарточные границы кратны плитке; последний кусок может быть короче.
    const int chunk = ((n + requested_gpus * tile - 1) /
                       (requested_gpus * tile)) * tile;
    const int used_gpus = (n + chunk - 1) / chunk;
    device_ids.resize(used_gpus);
    std::cout << "Merge-sort: " << used_gpus << " GPU(s), tile=" << tile
              << ", CUDA device IDs:";
    for (int d : device_ids) std::cout << ' ' << d;
    std::cout << std::endl;

    FastRandom random;
    std::vector<unsigned int> input(n);
    for (int i = 0; i < n; ++i) input[i] = random.next(1, max_value);
    if (n > 1) input[n - 1] = input[0];  // Гарантированно проверяем повтор значения.
    std::cout << "n=" << n << " values in range [1; " << max_value << "]" << std::endl;

    std::vector<unsigned int> expected = input;
    std::cout << "sorting on CPU..." << std::endl;
    timer cpu_timer;
    std::sort(expected.begin(), expected.end());
    const double cpu_elapsed = cpu_timer.elapsed();
    std::cout << "CPU std::sort finished in " << cpu_elapsed << " sec" << std::endl;

    Workspace workspace;
    workspace.gpus.resize(used_gpus);
    for (int k = 0; k < used_gpus; ++k) {
        auto& g = workspace.gpus[k];
        g.device = device_ids[k];
        g.offset = k * chunk;
        g.count = std::min(chunk, n - g.offset);
        allocate(g, k == 0 ? n : g.count);
    }
    for (int k = 1; k < used_gpus; ++k) {
        enable_peer(root_device, device_ids[k]);
        enable_peer(device_ids[k], root_device);
    }

    auto& root = workspace.gpus[0];
    const size_t bytes = size_t(n) * sizeof(unsigned int);
    MS_CUDA(cudaSetDevice(root_device));
    // Как и в исходном тесте: единственная H2D-загрузка вне замера сортировки.
    MS_CUDA(cudaMemcpyAsync(root.input, input.data(), bytes,
                           cudaMemcpyHostToDevice, root.stream));
    MS_CUDA(cudaStreamSynchronize(root.stream));

    int final_passes = 0;
    for (int width = chunk; width < n; width *= 2) ++final_passes;
    unsigned int* gathered = (final_passes & 1) ? root.temporary : root.output;
    unsigned int* other = (final_passes & 1) ? root.output : root.temporary;

    const auto run_sort = [&]() {
        if (used_gpus == 1) {
            MS_CUDA(cudaSetDevice(root_device));
            merge_fast::sort(root.input, root.output, root.temporary,
                             root.partitions, n, root.stream);
        } else {
            // Раздача -> локальная сортировка -> сбор. Все это входит в замер.
            // Worker streams работают независимо; CPU не ждет каждую карту отдельно.
            for (int k = 1; k < used_gpus; ++k) {
                auto& g = workspace.gpus[k];
                MS_CUDA(cudaSetDevice(g.device));
                const size_t part_bytes = size_t(g.count) * sizeof(unsigned int);
                MS_CUDA(cudaMemcpyPeerAsync(g.input, g.device,
                    root.input + g.offset, root_device, part_bytes, g.stream));
                merge_fast::sort(g.input, g.output, g.temporary,
                                 g.partitions, g.count, g.stream);
                MS_CUDA(cudaMemcpyPeerAsync(gathered + g.offset, root_device,
                    g.output, g.device, part_bytes, g.stream));
                MS_CUDA(cudaEventRecord(g.done, g.stream));
            }

            MS_CUDA(cudaSetDevice(root_device));
            merge_fast::sort(root.input, gathered, other, root.partitions,
                             root.count, root.stream);
            for (int k = 1; k < used_gpus; ++k) {
                MS_CUDA(cudaStreamWaitEvent(root.stream, workspace.gpus[k].done, 0));
            }

            // Финальные слияния на основной GPU. При четырех кусках это два этапа.
            unsigned int* src = gathered;
            unsigned int* dst = other;
            for (int width = chunk; width < n; width *= 2) {
                merge_fast::merge_pass(src, dst, root.partitions, width, n, root.stream);
                std::swap(src, dst);
            }
        }
        // Замер заканчивается только после получения полного результата на основной GPU.
        MS_CUDA(cudaStreamSynchronize(root.stream));
    };

    std::cout << "GPU warm-up..." << std::endl;
    run_sort();
    std::vector<double> times;
    times.reserve(iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        timer t;
        run_sort();
        times.push_back(t.elapsed());
    }
    std::cout << "GPU merge-sort times (in seconds) - "
              << stats::valuesStatsLine(times) << std::endl;
    const double median = stats::median(times);
    const double logical_gib = 2.0 * bytes / (1024.0 * 1024.0 * 1024.0);
    // Это прежняя условная метрика 2*n*sizeof(uint), НЕ реальный трафик всех этапов.
    std::cout << "GPU merge-sort median effective VRAM bandwidth: "
              << logical_gib / median << " GiB/s ("
              << n / 1e6 / median << " uint millions/s)" << std::endl;

    MS_CUDA(cudaSetDevice(root_device));
    std::vector<unsigned int> actual(n);
    MS_CUDA(cudaMemcpy(actual.data(), root.output, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error("Sort mismatch at i=" + std::to_string(i) +
                ": CPU=" + std::to_string(expected[i]) +
                ", GPU=" + std::to_string(actual[i]));
        }
    }
    // Переиспользуем CPU-буфер для проверки неизменности исходных данных.
    MS_CUDA(cudaMemcpy(actual.data(), root.input, bytes, cudaMemcpyDeviceToHost));
    if (actual != input) throw std::runtime_error("Input buffer was modified");
    std::cout << "OK: GPU result matches CPU; input is unchanged." << std::endl;
#endif
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit_code = 1;
    }
    avk2::InstanceContext::clearGlobalInstanceContext();
    return exit_code;
}
