#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки
#include "libgpu/work_size.h"

#include <fstream>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_radixMap(ocl::getRadixMap());
    ocl::KernelSource ocl_radixPut(ocl::getRadixPut());
    ocl::KernelSource ocl_prefixReduce(ocl::getPrefixReduction());
    ocl::KernelSource ocl_prefixAccum(ocl::getPrefixAccumulation());
    ocl::KernelSource ocl_prefixOneGroupReduce(ocl::getPrefixOneGroupReduction());
    ocl::KernelSource ocl_prefixOneGroupAccum(ocl::getPrefixOneGroupAccum());
    FastRandom r;

    // int n = 100*1000*1000;
    int n = 100*1000;
    int max_value = std::numeric_limits<int>::max();
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
    gpu::gpu_mem_32u input_gpu(n), output_gpu(n), output2_gpu(n);
    gpu::gpu_mem_32u scanner(2 * n);
    gpu::gpu_mem_32u group_accum(n >> GROUP_SHIFT);
    gpu::gpu_mem_32u mid_bucket_accum(2);
    input_gpu.fill(max_value);
    input_gpu.writeN(as.data(), n);
    output_gpu.fill(0);
    output2_gpu.fill(0);
    scanner.fill(0);
    group_accum.fill(0);
    mid_bucket_accum.fill(0);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        for (int radx = 0; radx < (32 / RADIX_SHIFT); ++radx) {
            for (int mask = 0; mask < (1 << RADIX_SHIFT); ++mask) {
                gpu::gpu_mem_32u& input = (radx == 0 ? input_gpu : (radx % 2 == 1 ? output2_gpu : output_gpu));
                gpu::gpu_mem_32u& output = (radx % 2 == 1 ? output_gpu : output2_gpu);

                int from = 0, to = n;
                int num = to - from;

                ocl_prefixOneGroupReduce.exec(
                        gpu::WorkSize(GROUP_SIZE, num),
                        scanner, input,
                        radx, mask,
                        from, to);
                num >>= GROUP_SHIFT;
                from = to;
                to = from + num;
                while (num > 1) {
                    gpu::WorkSize workSize(GROUP_SIZE, num);
                    ocl_prefixReduce.exec(workSize, scanner, from, to);
                    from = to;
                    num = (num + 1) / 2;
                    to += num;
                }

                ocl_prefixOneGroupAccum.exec(
                        gpu::WorkSize(GROUP_SIZE, n >> GROUP_SHIFT),
                        scanner, group_accum, n);

                ocl_radixPut.exec(
                        gpu::WorkSize(GROUP_SIZE, n),
                        input, output, scanner, group_accum, mid_bucket_accum,
                        n, radx, mask);
            }
        }

        times.push_back(t.elapsed());

    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = output_gpu.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

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
