#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

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

    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());
    ocl::KernelSource ocl_write_result(ocl::getPrefixSum03WriteResult());

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
    // Аллоцируем буферы в VRAM
    // Пирамида частичных сумм лежит в одном буфере: вершина (1 элемент) - в начале,
    // ниже - всё более длинные уровни; сумма длин уровней <= n + ceil(log2(n)) + 1
    gpu::gpu_mem_32u input_gpu(n), pow2_tree_gpu(n + 64), prefix_sum_accum_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Вершина пирамиды (tree[0]) - seed down-sweep: префикс перед первым блоком равен нулю.
    // Вверх пирамида вершину не строит, вниз никто в неё не пишет, поэтому нуля достаточно на все итерации
    const unsigned int zero = 0;
    pow2_tree_gpu.writeN(&zero, 1);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;
        // Раскладка пирамиды: длины уровней снизу вверх (m_1 = ceil(n/2), m_{k+1} = ceil(m_k/2), вершина - 1 элемент)
        std::vector<unsigned int> level_lens; // level_lens[i] - длина уровня i+1
        for (unsigned int m = (n + 1) / 2; ; m = (m + 1) / 2) {
            level_lens.push_back(m);
            if (m == 1) {
                break;
            }
        }
        // Смещения уровней: вершина - в начале буфера, каждый следующий (более длинный) уровень - ниже
        std::vector<unsigned int> level_offs(level_lens.size());
        {
            unsigned int tree_size = 0;
            for (size_t i = 0; i < level_lens.size(); ++i) {
                tree_size += level_lens[i];
            }
            rassert(tree_size <= n + 64, 53451234512341, tree_size, n);
            unsigned int tail = tree_size;
            for (size_t i = 0; i < level_lens.size(); ++i) {
                tail -= level_lens[i];
                level_offs[i] = tail; // уровень i+1 (длина level_lens[i]) начинается с этой позиции
            }
        }

        if (context.type() == gpu::Context::TypeOpenCL) {
            // Up-sweep: строим пирамиду частичных сумм снизу вверх;
            // нижний уровень читает входной массив напрямую, каждый запуск суммирует пары соседних элементов уровня.
            // Вершину (последний уровень из 1 элемента) не строим: scan'у нужна не полная сумма, а ноль перед первым блоком
            for (size_t i = 0; i + 1 < level_lens.size(); ++i) {
                const unsigned int src_len = (i == 0) ? n : level_lens[i - 1];
                const unsigned int src_off = (i == 0) ? 0 : level_offs[i - 1];
                const unsigned int dst_off = level_offs[i];
                const gpu::WorkSize ws(GROUP_SIZE, level_lens[i]);
                if (i == 0) {
                    ocl_sum_reduction.exec(ws, input_gpu, pow2_tree_gpu, src_off, dst_off, src_len);
                } else {
                    ocl_sum_reduction.exec(ws, pow2_tree_gpu, pow2_tree_gpu, src_off, dst_off, src_len);
                }
            }

            // Down-sweep: переписываем blocksum-ы каждого уровня на префиксы начал его блоков,
            // беря seed с уровня выше (уровни обрабатываем сверху вниз)
            for (size_t i = level_lens.size() - 1; i >= 1; --i) {
                const unsigned int m = level_lens[i - 1];      // длина текущего уровня
                const unsigned int level_off = level_offs[i - 1];
                const unsigned int coarse_off = level_offs[i]; // уровень выше уже содержит префиксы
                const gpu::WorkSize ws(GROUP_SIZE, (m + 1) / 2);
                ocl_prefix_accumulation.exec(ws, pow2_tree_gpu, level_off, coarse_off, m);
            }

            // Финальный шаг: префиксы перед парами + сами элементы => включительные префиксы
            ocl_write_result.exec(gpu::WorkSize(GROUP_SIZE, (n + 1) / 2), input_gpu, pow2_tree_gpu, prefix_sum_accum_gpu, level_offs[0], n);
        } else {
            rassert(false, 4531412341, context.type());
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
    std::vector<unsigned int> gpu_prefix_sum = prefix_sum_accum_gpu.readVector();
    std::cout << "GPU results read back" << std::endl;

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cpu_prefix_sum[i] == gpu_prefix_sum[i], 566324523452323, cpu_prefix_sum[i], gpu_prefix_sum[i], i);
    }
    std::cout << "GPU prefix sum verification passed" << std::endl;

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
    std::cout << "Input data verification passed" << std::endl;
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            exit_code = 0;
        } else {
            exit_code = 1;
        }
    }

    return exit_code;
}
