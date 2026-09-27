#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

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

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;
    const size_t n = size_t(width) * height;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * n / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> bs(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u a_gpu(n), b_gpu(n), c_gpu(n);

    a_gpu.writeN(as.data(), n);
    b_gpu.writeN(bs.data(), n);

    const gpu::WorkSize workSize(GROUP_SIZE, n);
    const double memory_size_gb = sizeof(unsigned int) * 3.0 * n / 1024.0 / 1024.0 / 1024.0;

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        // Запускаем кернел (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            ocl_aplusb_matrix_bad.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);

            times.push_back(t.elapsed());
        }
        std::cout << "BAD matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        std::cout << "BAD matrix kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        std::vector<unsigned int> cs(n, 0);
        c_gpu.readN(cs.data(), n);

        // Сверяем результат
        for (size_t i = 0; i < n; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    // Обнуляем выходной буфер, чтобы результат предыдущего кернела не мог замаскировать ошибку следующего
    c_gpu.fill(0);

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;
            ocl_aplusb_matrix_good.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            times.push_back(t.elapsed());
        }
        std::cout << "GOOD matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
        std::cout << "GOOD matrix kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        std::vector<unsigned int> cs(n, 0);
        c_gpu.readN(cs.data(), n);

        // Сверяем результат
        for (size_t i = 0; i < n; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230365731436, cs[i], as[i] + bs[i], i);
        }
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
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
