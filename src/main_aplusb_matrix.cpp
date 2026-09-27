#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <cstdlib>
#include <string>

struct Params {
    unsigned work_size_x;
    unsigned work_size_y;
    unsigned width;
    unsigned height;

    gpu::gpu_mem_32u a_gpu;
    gpu::gpu_mem_32u b_gpu;
    gpu::gpu_mem_32u c_gpu;
};

template <typename Shader>
std::vector<double> runVulkanShader(const Params& shader_params, Shader& shader) {
    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Настраиваем размер рабочего пространства (n) и размер рабочих групп в этом рабочем пространстве (GROUP_SIZE=256)
        // Обратите внимание что сейчас указана рабочая группа размера 1х1 в рабочем пространстве width x height, это не то что вы хотите
        gpu::WorkSize workSize(shader_params.work_size_x,
                               shader_params.work_size_y,
                               shader_params.width,
                               shader_params.height);

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        struct {
            unsigned int width;
            unsigned int height;
        } params = { shader_params.width, shader_params.height };
        shader.exec(params,
                    workSize,
                    shader_params.a_gpu,
                    shader_params.b_gpu,
                    shader_params.c_gpu);

        times.push_back(t.elapsed());
    }
    return times;
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 100 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 100 после этого реализуйте два кернела - максимально эффективный и максимально неэффктивный вариант сложения матриц - src/kernels/<ваш выбор>/aplusb_matrix_<bad/good>.<ваш выбор>
    // TODO 100 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DGPU_CUDA_SUPPORT=ON в CMake options
    gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(avk2::getAplusBMatrixBad());
    avk2::KernelSource vk_aplusb_matrix_good(avk2::getAplusBMatrixGood());

    Params shader_params;

    shader_params.work_size_x = GROUP_SIZE_X_BAD;
    shader_params.work_size_y = GROUP_SIZE_Y_BAD;

    unsigned task_size = 64;
    if (const char* requested_size = std::getenv("GPGPU_MATRIX_TASK_SIZE")) {
        const unsigned long parsed_size = std::stoul(requested_size);
        rassert(parsed_size >= 1 && parsed_size <= 64, 923854269);
        task_size = static_cast<unsigned>(parsed_size);
    }
    const unsigned width = shader_params.width = task_size * 256;
    const unsigned height = shader_params.height = task_size * 128;
    const unsigned size = width * height;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * size / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(size, 0);
    std::vector<unsigned int> bs(size, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    shader_params.a_gpu = gpu::gpu_mem_32u(size);
    shader_params.b_gpu = gpu::gpu_mem_32u(size);
    shader_params.c_gpu = gpu::gpu_mem_32u(size);

    shader_params.a_gpu.writeN(as.data(), size);
    shader_params.b_gpu.writeN(bs.data(), size);

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        const auto times = runVulkanShader(shader_params, vk_aplusb_matrix_bad);
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double memory_size_gb = sizeof(unsigned int) * 3 * size / 1024.0 / 1024.0 / 1024.0;
        std::cout << "a + b kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        std::vector<unsigned int> cs(size, 0);
        shader_params.c_gpu.readN(cs.data(), size);

        // Сверяем результат
        for (size_t i = 0; i < size; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    // Обнуляем выходной буфер, чтобы результат предыдущего кернела не мог замаскировать ошибку следующего
    shader_params.c_gpu.fill(0);

    shader_params.work_size_x = GROUP_SIZE_X_GOOD;
    shader_params.work_size_y = GROUP_SIZE_Y_GOOD;

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;

        const auto times = runVulkanShader(shader_params, vk_aplusb_matrix_good);
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double memory_size_gb = sizeof(unsigned int) * 3 * size / 1024.0 / 1024.0 / 1024.0;
        std::cout << "a + b kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        std::vector<unsigned int> cs(size, 0);
        shader_params.c_gpu.readN(cs.data(), size);

        // Сверяем результат
        for (size_t i = 0; i < size; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
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
