#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iostream>
#include <vector>

void validateResult(
    gpu::gpu_mem_32u& c_gpu,
    const std::vector<unsigned int>& a,
    const std::vector<unsigned int>& b,
    unsigned int width,
    unsigned int height)
{
    const size_t cell_count =
        static_cast<size_t>(width) * height;

    std::vector<unsigned int> result(cell_count);
    c_gpu.readN(result.data(), cell_count);

    for (size_t i = 0; i < cell_count; ++i) {
        const unsigned int expected = a[i] + b[i];

        rassert(
            result[i] == expected,
            321418230421312512,
            result[i],
            expected,
            i);
    }
}

void run_sum(
    gpu::WorkSize work_size,
    ocl::KernelSource& kernel,
    gpu::Context& context,
    gpu::gpu_mem_32u& a_gpu,
    gpu::gpu_mem_32u& b_gpu,
    gpu::gpu_mem_32u& c_gpu,
    const std::vector<unsigned int>& a,
    const std::vector<unsigned int>& b,
    unsigned int width,
    unsigned int height,
    const char* state)
{
    const size_t cell_count =
        static_cast<size_t>(width) * height;

    std::cout << "Running " << state << " matrix kernel...\n";

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        if (context.type() == gpu::Context::TypeOpenCL) {
            kernel.exec(
                work_size,
                a_gpu,
                b_gpu,
                c_gpu,
                width,
                height);
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }

    std::cout << "a + b matrix " << state
              << " kernel times (in seconds) - "
              << stats::valuesStatsLine(times) << '\n';

    const double memory_size_gb =
        sizeof(unsigned int) * 3.0 * cell_count
        / 1024.0 / 1024.0 / 1024.0;

    std::cout << "a + b matrix " << state
              << " kernel median VRAM bandwidth: "
              << memory_size_gb / stats::median(times)
              << " GiB/s\n";

    validateResult(c_gpu, a, b, width, height);
}

void run(int argc, char** argv)
{
    gpu::Device device =
        gpu::chooseGPUDevice(
            gpu::selectAllDevices(ALL_GPUS, true),
            argc,
            argv);

    gpu::Context context =
        activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_aplusb_matrix_bad(
        ocl::getAplusBMatrixBad());

    ocl::KernelSource ocl_aplusb_matrix_good(
        ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(
        avk2::getAplusBMatrixBad());

    avk2::KernelSource vk_aplusb_matrix_good(
        avk2::getAplusBMatrixGood());

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;

    const size_t matrix_cell_cnt =
        static_cast<size_t>(width) * height;

    std::cout << "matrices size: "
              << width << "x" << height
              << " = 3 * "
              << (sizeof(unsigned int) * 3.0 * matrix_cell_cnt
                  / 1024.0 / 1024.0)
              << " MiB\n";

    std::vector<unsigned int> as(matrix_cell_cnt, 0);
    std::vector<unsigned int> bs(matrix_cell_cnt, 0);

    for (size_t i = 0; i < matrix_cell_cnt; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    // Аллоцируем буферы в VRAM.
    gpu::gpu_mem_32u a_gpu(matrix_cell_cnt);
    gpu::gpu_mem_32u b_gpu(matrix_cell_cnt);
    gpu::gpu_mem_32u c_gpu(matrix_cell_cnt);

    a_gpu.writeN(as.data(), matrix_cell_cnt);
    b_gpu.writeN(bs.data(), matrix_cell_cnt);

    run_sum(
        gpu::WorkSize(256, 1, width, height),
        ocl_aplusb_matrix_bad,
        context,
        a_gpu, b_gpu, c_gpu,
        as, bs,
        width, height,
        "BAD");

    c_gpu.fill(0);

    run_sum(
        gpu::WorkSize(16, 16, width, height),
        ocl_aplusb_matrix_good,
        context,
        a_gpu, b_gpu, c_gpu,
        as, bs,
        width, height,
        "GOOD");
}

int main(int argc, char** argv)
{
    int exit_code = 0;

    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;

        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // API не поддерживается на устройстве, например в CI.
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Задание ещё не реализовано.
            exit_code = 0;
        } else {
            exit_code = 1;
        }
    }

    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
