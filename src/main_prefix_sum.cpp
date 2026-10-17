#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/cuda/utils.h>
#include <libgpu/vulkan/engine.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

void run(int argc, char** argv)
{
    int num_devices = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        throw std::runtime_error(DEVICE_NOT_SUPPORT_API);
    }

    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    std::vector<unsigned int> cpu_prefix_sum(n);
    {
        std::cout << "prefix sum on CPU..." << std::endl;
        timer t;
        unsigned int sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum += as[i];
            cpu_prefix_sum[i] = sum;
        }
        double elapsed = t.elapsed();
        double memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU prefix sum finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU prefix sum effective RAM bandwidth: "
                  << memory_size_gb / elapsed << " GB/s ("
                  << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    unsigned int chunk_size = n / num_devices / 4 * 4;
    std::vector<unsigned int> counts(num_devices, chunk_size);
    counts.back() += n - chunk_size * num_devices;
    std::vector<unsigned int*> inputs(num_devices);
    std::vector<unsigned int*> outputs(num_devices);
    for (int device = 0; device < num_devices; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&inputs[device]), sizeof(unsigned int) * counts[device]));
        CUDA_SAFE_CALL(cudaMemcpy(inputs[device], as.data() + device * chunk_size, sizeof(unsigned int) * counts[device], cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&outputs[device]), sizeof(unsigned int) * counts[device]));
    }

    // Warmup and lazy initialization
    cuda::prefix_sum_02_prefix_accumulation(num_devices, counts, inputs, outputs);
    for (int device = 0; device < num_devices; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        cuda::prefix_sum_02_prefix_accumulation(num_devices, counts, inputs, outputs);

        for (int device = 0; device < num_devices; ++device) {
            CUDA_SAFE_CALL(cudaSetDevice(device));
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили префиксные суммы)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU prefix sum median effective VRAM bandwidth: "
              << memory_size_gb / stats::median(times) << " GB/s ("
              << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_prefix_sum(n);
    for (int device = 0; device < num_devices; ++device) {
        CUDA_SAFE_CALL(cudaSetDevice(device));
        CUDA_SAFE_CALL(cudaMemcpy(gpu_prefix_sum.data() + device * chunk_size, outputs[device],
            sizeof(unsigned int) * counts[device], cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaFree(inputs[device]));
        CUDA_SAFE_CALL(cudaFree(outputs[device]));
    }

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cpu_prefix_sum[i] == gpu_prefix_sum[i], 566324523452323, cpu_prefix_sum[i], gpu_prefix_sum[i], i);
    }
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            exit_code = 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
