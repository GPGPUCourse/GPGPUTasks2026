#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <vector>

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
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);

    a_gpu.writeN(as.data(), as.size());
    b_gpu.writeN(bs.data(), bs.size());

    auto benchmark = [&](const char* name, ocl::KernelSource& kernel,
                         const gpu::WorkSize& work_size) {
        std::cout << "Running " << name << " matrix kernel..." << std::endl;

        kernel.exec(work_size, a_gpu, b_gpu, c_gpu, width, height);
        c_gpu.fill(0);

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;
            kernel.exec(work_size, a_gpu, b_gpu, c_gpu, width, height);
            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - "
                  << stats::valuesStatsLine(times) << std::endl;

        const double bytes = 3.0 * sizeof(unsigned int) * width * height;
        const double median_time = stats::median(times);
        std::cout << "a + b matrix kernel median VRAM bandwidth: "
                  << bytes / median_time / 1e9 << " GB/s" << std::endl;

        std::vector<unsigned int> cs(as.size());
        c_gpu.readN(cs.data(), cs.size());
        for (size_t i = 0; i < cs.size(); ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512,
                    cs[i], as[i] + bs[i], i);
        }
        std::cout << name << " result: OK" << std::endl;
        return median_time;
    };

    const double bad_time = benchmark("BAD", ocl_aplusb_matrix_bad,
                                     gpu::WorkSize(GROUP_SIZE, 1, height, width));
    const double good_time = benchmark("GOOD", ocl_aplusb_matrix_good,
                                      gpu::WorkSize(GROUP_SIZE, 1, width, height));
    std::cout << "GOOD speedup over BAD: " << bad_time / good_time << "x" << std::endl;
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
