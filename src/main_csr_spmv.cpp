#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <algorithm>
#include <fstream>

std::tuple<std::vector<unsigned int>, std::vector<unsigned int>, std::vector<unsigned int>> generate_csr_matrix(
    unsigned int nrows, unsigned int ncols,
    unsigned int min_non_zero_values_per_row, unsigned int max_non_zero_values_per_row,
    unsigned int max_value)
{
    std::vector<unsigned int> csr_row_offsets(nrows + 1, 0);
    for (unsigned int row = 0; row < nrows; ++row) {
        FastRandom r(239 * row);
        unsigned int non_zero_row_values = r.next(min_non_zero_values_per_row, max_non_zero_values_per_row);
        csr_row_offsets[row + 1] = csr_row_offsets[row] + non_zero_row_values;
    }
    const unsigned int number_of_non_zero_values = csr_row_offsets[nrows];

    std::vector<unsigned int> csr_columns(number_of_non_zero_values, 0);
    std::vector<unsigned int> csr_values(number_of_non_zero_values, 0);

    #pragma omp parallel for
    for (int row = 0; row < nrows; ++row) {
        FastRandom r(239 * row);
        unsigned int non_zero_row_values = csr_row_offsets[row + 1] - csr_row_offsets[row];
        std::vector<unsigned int> columns = r.random_sorted_unique_values(0, ncols - 1, non_zero_row_values);

        for (int i = 0; i < non_zero_row_values; ++i) {
            unsigned int value = r.next(0, max_value);
            csr_columns[csr_row_offsets[row] + i] = columns[i];
            csr_values[csr_row_offsets[row] + i] = value;
        }
    }

    return {csr_row_offsets, csr_columns, csr_values};
}

std::vector<unsigned int> generate_vector(
    unsigned int n, unsigned int max_value=10000)
{
    FastRandom r(2391);

    std::vector<unsigned int> data(n, 0);
    for (unsigned int i = 0; i < n; ++i) {
        data[i] = r.next(0, max_value);
    }

    return data;
}

std::vector<unsigned int> sparse_csr_matrix_vector_multiplication(
    const std::vector<unsigned int> &csr_row_offsets, const std::vector<unsigned int> &csr_columns, const std::vector<unsigned int> &csr_values,
    const std::vector<unsigned int> &vector_values,
    unsigned int nrows, unsigned int ncols)
{
    std::vector<unsigned int> result_vector_values(nrows, 0);

    #pragma omp parallel for schedule(dynamic, 128)
    for (int row = 0; row < nrows; ++row) {
        size_t accumulator = 0;
        unsigned int row_from = csr_row_offsets[row];
        unsigned int row_to = csr_row_offsets[row + 1];
        for (unsigned int i = row_from; i < row_to; ++i) {
            unsigned int col = csr_columns[i];
            accumulator += csr_values[i] * vector_values[col];
        }
        result_vector_values[row] = narrow_cast<unsigned int>(accumulator);
    }

    return result_vector_values;
}

#ifdef CUDA_SUPPORT
cuda::SpmvConfig select_spmv_config(
    const std::vector<unsigned int>& row_offsets,
    const std::vector<unsigned int>& nnz_per_row,
    const gpu::gpu_mem_32u& row_offsets_gpu,
    const gpu::gpu_mem_32u& columns_gpu,
    const gpu::gpu_mem_32u& values_gpu,
    const gpu::gpu_mem_32u& x_gpu,
    gpu::gpu_mem_32u& y_gpu,
    unsigned int nrows)
{
    timer tuning_timer;
    unsigned int fixed_nnz = 0;
    if (nrows != 0 && row_offsets.front() == 0) {
        const unsigned int first = nnz_per_row.front();
        if ((first == 32 || first == 128)
            && std::all_of(nnz_per_row.begin(), nnz_per_row.end(),
                [first](unsigned int n) { return n == first; })) {
            fixed_nnz = first;
        }
    }

    std::vector<cuda::SpmvConfig> candidates = {{32, 256, 0, false, true}};
    for (unsigned int width : {8u, 16u, 32u}) {
        for (unsigned int block : {128u, 256u, 512u}) {
            for (bool streaming : {false, true}) {
                candidates.push_back({width, block, 0, streaming, false});
                if (fixed_nnz != 0) {
                    candidates.push_back({width, block, fixed_nnz, streaming, false});
                }
            }
        }
    }

    const auto launch = [&](const cuda::SpmvConfig& config) {
        cuda::sparse_csr_matrix_vector_multiplication(
            config, row_offsets_gpu, columns_gpu, values_gpu, x_gpu, y_gpu, nrows);
    };

    std::vector<std::vector<double>> samples(candidates.size());
    for (auto& values : samples) {
        values.reserve(5);
    }

    for (int round = 0; round < 5; ++round) {
        for (size_t step = 0; step < candidates.size(); ++step) {
            const size_t i = (round % 2 == 0) ? step : candidates.size() - 1 - step;
            launch(candidates[i]);
            timer measurement;
            launch(candidates[i]);
            samples[i].push_back(measurement.elapsed());
        }
    }

    size_t best = 0;
    double best_time = stats::median(samples[0]);
    for (size_t i = 1; i < candidates.size(); ++i) {
        const double elapsed = stats::median(samples[i]);
        if (elapsed < best_time) {
            best = i;
            best_time = elapsed;
        }
    }

    const auto config = candidates[best];
    std::cout << "SpMV selected: " << (config.baseline ? "baseline" : "optimized")
              << ", threads/row=" << config.threads_per_row
              << ", block=" << config.block_size
              << ", fixed_nnz=" << config.fixed_nnz_per_row
              << ", matrix_loads=" << (config.baseline ? "default"
                  : (config.streaming_loads ? "streaming" : "read-only"))
              << std::endl;
    std::cout << "SpMV tuning: " << tuning_timer.elapsed() << " sec, "
              << candidates.size() << " candidates, baseline="
              << stats::median(samples[0]) * 1000.0 << " ms, selected="
              << best_time * 1000.0 << " ms" << std::endl;
    return config;
}
#endif

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);

    ocl::KernelSource ocl_spvm(ocl::getSparseCSRMatrixVectorMult());

    avk2::KernelSource vk_spvm(avk2::getSparseCSRMatrixVectorMult());

    FastRandom r;

    const unsigned int nrows = 1000*1000;
    const unsigned int ncols = 1000*1000;
    const unsigned int max_value = 1000;
    std::cout << "Evaluating CSR matrix nrows x ncols=" << nrows << "x" << ncols << " with values in range [0; " << max_value << "]" << std::endl;

    std::vector<std::pair<unsigned int, unsigned int>> evaluated_min_max_nnz_per_row = {
        {32, 32},
        {128, 128},
        {1, 32},
        {1, 128},
        {32, 128},
    };
    double cpu_bandwidth_sum = 0.0, gpu_bandwidth_sum = 0.0;
    double cpu_throughput_sum = 0.0, gpu_throughput_sum = 0.0;
    for (auto [min_nnz_per_row, max_nnz_per_col]: evaluated_min_max_nnz_per_row) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        const auto [csr_row_offsets, csr_columns, csr_values] = generate_csr_matrix(nrows, ncols, min_nnz_per_row, max_nnz_per_col, max_value);
        const std::vector<unsigned int> vector_values = generate_vector(ncols, max_value);
        const unsigned int number_of_non_zero_values = csr_row_offsets[nrows];
        const unsigned int nnz = number_of_non_zero_values;

        std::vector<unsigned int> nnz_per_row(nrows);
        for (int row = 0; row < nrows; ++row) {
            nnz_per_row[row] = csr_row_offsets[row + 1] - csr_row_offsets[row];
        }

        std::cout << "Evaluating with NNZ per row in range [" << min_nnz_per_row << "; " << max_nnz_per_col << "], median NNZ per row=" << stats::median(nnz_per_row) << ", total NNZ=" << nnz << "..." << std::endl;

        timer t;
        const std::vector<unsigned int> cpu_results = sparse_csr_matrix_vector_multiplication(csr_row_offsets, csr_columns, csr_values, vector_values, nrows, ncols);
        double elapsed = t.elapsed();
        double memory_size_gb = sizeof(unsigned int) * (nnz + vector_values.size() + cpu_results.size()) / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU (multi-threaded via OpenMP) finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU effective bandwidth: " << memory_size_gb / elapsed << " GB/s (" << nnz / 1e6 / elapsed << " uint millions/s)" << std::endl;

        gpu::gpu_mem_32u csr_row_offsets_gpu(nrows + 1), csr_columns_gpu(nnz), csr_values_gpu(nnz), vector_values_gpu(ncols), output_vector_values_gpu(nrows);

        csr_row_offsets_gpu.writeN(csr_row_offsets.data(), csr_row_offsets.size());
        csr_columns_gpu.writeN(csr_columns.data(), csr_columns.size());
        csr_values_gpu.writeN(csr_values.data(), csr_values.size());
        vector_values_gpu.writeN(vector_values.data(), vector_values.size());

#ifdef CUDA_SUPPORT
        const cuda::SpmvConfig spmv_config = select_spmv_config(
            csr_row_offsets, nnz_per_row, csr_row_offsets_gpu, csr_columns_gpu,
            csr_values_gpu, vector_values_gpu, output_vector_values_gpu, nrows);
#endif

        output_vector_values_gpu.fill(255);

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            t.restart();

            if (context.type() == gpu::Context::TypeOpenCL) {
                throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            } else if (context.type() == gpu::Context::TypeCUDA) {
#ifdef CUDA_SUPPORT
                cuda::sparse_csr_matrix_vector_multiplication(
                    spmv_config,
                    csr_row_offsets_gpu,
                    csr_columns_gpu,
                    csr_values_gpu,
                    vector_values_gpu,
                    output_vector_values_gpu,
                    nrows);
#else
                throw std::runtime_error(DEVICE_NOT_SUPPORT_API);
#endif
            } else if (context.type() == gpu::Context::TypeVulkan) {
                throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }

        std::cout << "GPU SpMV (sparse matrix-vector multiplication) times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double gpu_elapsed = stats::median(times);
        std::cout << "GPU SpMV median effective VRAM bandwidth: " << memory_size_gb / gpu_elapsed << " GB/s (" << nnz / 1e6 / gpu_elapsed << " uint millions/s)" << std::endl;

        std::vector<unsigned int> gpu_results = output_vector_values_gpu.readVector();

        rassert(cpu_results.size() == gpu_results.size(), 43572348952, cpu_results.size(), gpu_results.size());

        for (size_t i = 0; i < cpu_results.size(); ++i) {
            rassert(cpu_results[i] == gpu_results[i], 566324523452323, cpu_results[i], gpu_results[i], i);
        }

        cpu_bandwidth_sum += memory_size_gb / elapsed;
        gpu_bandwidth_sum += memory_size_gb / gpu_elapsed;
        cpu_throughput_sum += nnz / 1e6 / elapsed;
        gpu_throughput_sum += nnz / 1e6 / gpu_elapsed;
    }

    const size_t count = evaluated_min_max_nnz_per_row.size();
    std::cout << "____________________________________________________________________________________________" << std::endl;
    std::cout << "Average SpMV performance across " << count << " matrix configurations:" << std::endl;
    std::cout << "CPU mean effective RAM bandwidth: " << cpu_bandwidth_sum / count << " GB/s ("
              << cpu_throughput_sum / count << " uint millions/s)" << std::endl;
    std::cout << "GPU mean effective VRAM bandwidth (based on per-configuration median times): "
              << gpu_bandwidth_sum / count << " GB/s ("
              << gpu_throughput_sum / count << " uint millions/s)" << std::endl;
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
