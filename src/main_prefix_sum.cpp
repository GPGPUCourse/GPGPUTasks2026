#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <vector>

void run(int argc, char** argv, unsigned int n = 100 * 1000 * 1000)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DGPU_CUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

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

    gpu::gpu_mem_32u input_gpu(n);
    std::vector<gpu::gpu_mem_32u> sums{input_gpu};
    while (sums.back().number() > 1) {
        const size_t next_n = (sums.back().number() + 1) / 2;
        sums.emplace_back(next_n);
    }
    std::vector<gpu::gpu_mem_32u> prefixes;
    for (size_t level = 0; level + 1 < sums.size(); ++level) {
        prefixes.emplace_back(sums[level].number());
    }
    prefixes.push_back(sums.back());

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        for (size_t level = 0; level + 1 < sums.size(); ++level) {
            const unsigned int level_n = static_cast<unsigned int>(sums[level].number());
            gpu::WorkSize work_size(GROUP_SIZE, (level_n + 1) / 2);
            ocl_sum_reduction.exec(work_size, sums[level], sums[level + 1], level_n);
        }
        for (size_t level = sums.size() - 1; level > 0; --level) {
            const unsigned int level_n = static_cast<unsigned int>(sums[level - 1].number());
            gpu::WorkSize work_size(GROUP_SIZE, (level_n + 1) / 2);
            ocl_prefix_accumulation.exec(work_size, sums[level - 1], prefixes[level], prefixes[level - 1], level_n);
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
    std::vector<unsigned int> gpu_prefix_sum = prefixes.front().readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cpu_prefix_sum[i] == gpu_prefix_sum[i], 566324523452323, cpu_prefix_sum[i], gpu_prefix_sum[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
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
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            exit_code = 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
