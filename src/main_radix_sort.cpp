#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());
    ocl::KernelSource ocl_radixSort04Scatter4(ocl::getRadixSort04Scatter4());

    FastRandom r;

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

    {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double elapsed = t.elapsed();
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / elapsed << " GB/s (" << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer1_gpu(n); // TODO это просто шаблонка, можете переименовать эти буферы, сделать другого размера/типа, удалить часть, добавить новые
    gpu::gpu_mem_32u buffer_output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    
    gpu::WorkSize workSize(GROUP_SIZE, n);
    
    input_gpu.writeN(as.data(), n);
    const unsigned int groups = div_ceil((unsigned int) n, (unsigned int) GROUP_SIZE);
    const unsigned int m = (1u << 4) * groups;
    const unsigned int level_size = div_ceil(m, (unsigned int) GROUP_SIZE);
    gpu::gpu_mem_32u counts_gpu(m), pref_gpu(m);
    gpu::gpu_mem_32u level1_gpu(level_size), level2_gpu(level_size), offsets_gpu(level_size);

    auto pref_sum = [&](uint m) {
        ocl_radixSort02GlobalPrefixesScanSumReduction.exec(gpu::WorkSize(GROUP_SIZE, m), counts_gpu, level1_gpu, pref_gpu, m, 0u);
        gpu::gpu_mem_32u *in = &level1_gpu, *out = &level2_gpu;
        unsigned int k = div_ceil(m, (unsigned int) GROUP_SIZE);
        for (unsigned int level = 1; k > 1; ++level, k = div_ceil(k, (unsigned int) GROUP_SIZE)) {
            ocl_radixSort02GlobalPrefixesScanSumReduction.exec(gpu::WorkSize(GROUP_SIZE, k), *in, *out, offsets_gpu, k, 0u);
            ocl_radixSort03GlobalPrefixesScanAccumulation.exec(gpu::WorkSize(GROUP_SIZE, m), offsets_gpu, pref_gpu, m, level);
            std::swap(in, out);
        }
    };

    for (int bits = 1; bits <= 4; ++bits) {
        // Запускаем кернел (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        auto Scatter = (bits == 4 ? ocl_radixSort04Scatter4 : ocl_radixSort04Scatter);
        for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
            timer t;
    
            // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
            // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
            const unsigned int passes = div_ceil(32u, (unsigned int) bits);
            gpu::gpu_mem_32u* src = &input_gpu;
            for (unsigned int offset = 0, pass = 0; offset < 32; offset += bits, ++pass) {
                gpu::gpu_mem_32u* dst = ((passes - pass) % 2 == 1) ? &buffer_output_gpu : &buffer1_gpu;
                ocl_radixSort01LocalCounting.exec(workSize, *src, counts_gpu, bits, offset, (unsigned int) n);
                pref_sum((1u << bits) * groups);
                Scatter.exec(workSize, *src, pref_gpu, *dst, bits, offset, (unsigned int) n);
                src = dst;
            }
            rassert(src == &buffer_output_gpu, 7345234512);
    
            times.push_back(t.elapsed());
        }
        std::cout << "GPU " << bits << "-bit-radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
    
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "GPU " << bits << "-bit-radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;
    
        // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
        std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();
    
        // Сверяем результат
        for (size_t i = 0; i < n; ++i) {
            rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
        }
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
