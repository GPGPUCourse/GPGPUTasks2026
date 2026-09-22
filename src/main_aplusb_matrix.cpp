#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <cstdint>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);
    /*
    // export VK_LAYER_PATH=/usr/local/share/vulkan/explicit_layer.d
    gpu::Context context;
    context.initVulkan(device.device_id_vulkan);
    context.setVKValidationLayers(true);
    context.activate();
    */

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(avk2::getAplusBMatrixBad());
    avk2::KernelSource vk_aplusb_matrix_good(avk2::getAplusBMatrixGood());

    uint32_t task_size = 64;
    uint32_t width = task_size * 256;
    uint32_t height = task_size * 128;
    uint32_t total = width * height;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * total / 1024 / 1024) << " MB\n";

    std::vector<unsigned int> as(total, 0);
    std::vector<unsigned int> bs(total, 0);
    for (size_t i = 0; i < total; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    gpu::gpu_mem_32u a_gpu(total), b_gpu(total), c_gpu(total);

    a_gpu.writeN(as.data(), total);
    b_gpu.writeN(bs.data(), total);

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(1, GROUP_SIZE, 1, height);

            if (context.type() == gpu::Context::TypeOpenCL) {
                // ocl_aplusb_matrix_bad.exec(workSize, a_gpu, ...);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                // cuda::aplusb_matrix_bad(workSize, a_gpu, ...);
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    uint32_t width;
                    uint32_t height;
                } params = { width, height };
                vk_aplusb_matrix_bad.exec(params, workSize, a_gpu, b_gpu, c_gpu);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
        double memory_size_gb = sizeof(unsigned int) * 3 * total / 1024.0 / 1024.0 / 1024.0;
        std::cout << "a + b matrix kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s\n";

        std::vector<unsigned int> cs(total, 0);
        c_gpu.readN(cs.data(), total);

        for (size_t i = 0; i < total; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    c_gpu.fill(0);

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(GROUP_SIZE, width);

            if (context.type() == gpu::Context::TypeOpenCL) {
                // ocl_aplusb_matrix_good.exec(workSize, a_gpu, ...);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                // cuda::aplusb_matrix_good(workSize, a_gpu, ...);
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    uint32_t width;
                    uint32_t height;
                } params = { width, height };
                vk_aplusb_matrix_good.exec(params, workSize, a_gpu, b_gpu, c_gpu);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;
        double memory_size_gb = sizeof(unsigned int) * 3 * total / 1024.0 / 1024.0 / 1024.0;
        std::cout << "a + b matrix kernel median VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s\n";

        std::vector<unsigned int> cs(total, 0);
        c_gpu.readN(cs.data(), total);

        for (size_t i = 0; i < total; ++i) {
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
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            exit_code = 0;
        } else {
            exit_code = 1;
        }
    }

    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
