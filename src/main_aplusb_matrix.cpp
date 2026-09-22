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

    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(avk2::getAplusBMatrixBad());
    avk2::KernelSource vk_aplusb_matrix_good(avk2::getAplusBMatrixGood());

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);

    a_gpu.writeN(as.data(), width * height);
    b_gpu.writeN(bs.data(), width * height);

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        // Запускаем кернел (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(GROUP_SIZE_Y, GROUP_SIZE_X, height, width);

            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_bad.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                cuda::aplusb_matrix_bad(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    unsigned int width;
                    unsigned int height;
                } params = { width, height };
                vk_aplusb_matrix_bad.exec(params, workSize, a_gpu, b_gpu, c_gpu);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix bad kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double memory_size_gb = sizeof(unsigned int) * 3.0 * width * height / 1024.0 / 1024.0 / 1024.0;
        std::cout << "a + b matrix bad kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.readN(cs.data(), width * height);

        // Сверяем результат
        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    // Обнуляем выходной буфер, чтобы результат предыдущего кернела не мог замаскировать ошибку следующего
    c_gpu.fill(0);

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;

        // Почти тот же код что с плохим кернелом, но теперь с хорошим, рекомендуется копи-паста
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(GROUP_SIZE_X, GROUP_SIZE_Y, width, height);

            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_good.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                cuda::aplusb_matrix_good(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    unsigned int width;
                    unsigned int height;
                } params = { width, height };
                vk_aplusb_matrix_good.exec(params, workSize, a_gpu, b_gpu, c_gpu);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix good kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double memory_size_gb = sizeof(unsigned int) * 3.0 * width * height / 1024.0 / 1024.0 / 1024.0;
        std::cout << "a + b matrix good kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.readN(cs.data(), width * height);

        // Сверяем результат
        for (size_t i = 0; i < width * height; ++i) {
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
