#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

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

    // Выбор API - CUDA (см. TODO 000 в шаблоне): у нас NVIDIA видеокарта, CUDA SDK установлен,
    // кернелы лежат в src/kernels/cu/*.cu, в CMake options выставлены -DGPU_CUDA_SUPPORT=ON,
    // -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA

    unsigned int n = 100 * 1000 * 1000;
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
    // input_gpu            - вход (не изменяется на протежении всего прогона)
    // tile_scan_gpu        - временный: inclusive-скан каждого тайла внутри себя (шаг 1)
    // prefix_sum_accum_gpu - результат: полная префиксная сумма (шаг 5)
    gpu::gpu_mem_32u input_gpu(n), tile_scan_gpu(n), prefix_sum_accum_gpu(n);

    // Вспомогательные буферы скана сумм тайлов (шаги 2-4).
    // Сумм тайлов n0 = ceil(n/SCAN_TILE) штук (для n = 100M это 97657),
    // и этот массив тоже сканируем тем же способом: n0 -> n1 -> n2, пока не влезет в один тайл.
    const unsigned int n0 = (n + SCAN_TILE - 1) / SCAN_TILE;
    const unsigned int n1 = (n0 + SCAN_TILE - 1) / SCAN_TILE;
    const unsigned int n2 = (n1 + SCAN_TILE - 1) / SCAN_TILE;
    // Двух уровней рекурсии достаточно пока n <= SCAN_TILE^3 (~1.07e9) - для задания (n = 100M) с запасом
    rassert(n <= SCAN_TILE * SCAN_TILE * SCAN_TILE, 567324523423, n, SCAN_TILE);
    gpu::gpu_mem_32u totals0(n0), scanned0(n0), // уровень 0: суммы тайлов входа и их полный префикс
        totals1(n1), scanned1(n1), // уровень 1: суммы тайлов уровня 0 и их полный префикс
        totals2(n2); // уровень 2 (одно число, нужно ядру 01 на уровне 1)

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Обнуляем вспомогательные буферы кернелом заполнения нулями
    auto zero_buffer = [](gpu::gpu_mem_32u& buffer) {
        const unsigned int size = buffer.number();
        cuda::fill_buffer_with_zeros(gpu::WorkSize(GROUP_SIZE, size), buffer, size);
    };

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        if (context.type() == gpu::Context::TypeCUDA) {
            // Обнуляем вспомогательные буферы кернелом заполнения нулями
            zero_buffer(totals0);
            zero_buffer(scanned0);
            zero_buffer(totals1);
            zero_buffer(scanned1);
            zero_buffer(totals2);

            // Шаг 1: независимый скан каждого тайла из SCAN_TILE элементов входа
            //   tile_scan[i] = sum input[base .. i] (внутри тайла), totals0[tile] = сумма тайла
            // Чтение n + запись n
            cuda::prefix_sum_01_reduction(input_gpu, tile_scan_gpu, totals0, n);

            // Шаги 2-4: скан сумм тайлов тем же способом (массивы маленькие, n0=97657 -> n1=96 -> n2=1)
            //   scanned0[tile] = сумма всех тайлов 0..tile
            cuda::prefix_sum_01_reduction(totals0, scanned0, totals1, n0); // скан тайлов totals0
            cuda::prefix_sum_01_reduction(totals1, scanned1, totals2, n1); // n1=96 <= SCAN_TILE: scanned1 уже полный скан
            cuda::prefix_sum_02_prefix_accumulation(scanned0, scanned1, scanned0, n0); // суммы тайлей слева для уровня 0

            // Шаг 5: наложение сумм предыдущих тайлов
            //   prefix_sum_accum[i] = tile_scan[i] + сумма всех тайлов слева = sum input[0 .. i]
            // Чтение n + запись n
            cuda::prefix_sum_02_prefix_accumulation(tile_scan_gpu, scanned0, prefix_sum_accum_gpu, n);
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

    // Сверяем результат с CPU вариантом
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
