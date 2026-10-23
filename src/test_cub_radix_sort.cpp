#include <libutils/misc.h>

#include "kernels/kernels.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
void run_tests(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = gpu::activateContext(device, gpu::Context::TypeCUDA);
    std::mt19937 random(0x5C0B2026u);
    const uint32_t max_key = std::numeric_limits<uint32_t>::max();
    const uint32_t boundaries[] = {0, 1, 255, 256, 0x7fffffffu, 0x80000000u, 0xfffffffeu, max_key};
    size_t cases = 0;

    for (int n : {0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 4097, 65537, 1000003}) {
        gpu::gpu_mem_32u input(n), output(n);
        gpu::gpu_mem_8u scratch(cuda::cub_radix_sort_temp_bytes(input, output, n));

        // Reuse the same allocations for unrelated inputs, catching stale state
        // and cached-result bugs. Every invocation is validated independently.
        for (int pattern = 0; pattern < 8; ++pattern) {
            std::vector<uint32_t> values(n);
            for (auto& value : values) {
                value = random(); // unrestricted 32-bit keys
                if (pattern == 1) value = 0;
                if (pattern == 2) value = max_key;
                if (pattern == 3) value %= 17;
                if (pattern == 4) value = boundaries[value % 8];
            }
            if (pattern == 5 || pattern == 6) {
                std::sort(values.begin(), values.end());
                if (pattern == 6) std::reverse(values.begin(), values.end());
            }
            auto expected = values;
            std::sort(expected.begin(), expected.end());
            if (n > 0) {
                input.writeN(values.data(), n);
                output.fill(0xdeadbeefu);
            }
            cuda::cub_radix_sort(input, output, scratch, n);
            if (n > 0) {
                rassert(output.readVector() == expected, 82050101, n, pattern);
                rassert(input.readVector() == values, 82050102, n, pattern);
            }
            ++cases;
        }
    }
    std::cout << "CUB radix-sort tests passed: " << cases
              << " cases (including empty, singleton, full uint32 range, duplicates,"
                 " high bits, sorted/reversed and nonmultiple sizes); input unchanged." << std::endl;
}
} // namespace

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run_tests(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit_code = 1;
    }
    avk2::InstanceContext::clearGlobalInstanceContext();
    return exit_code;
}
