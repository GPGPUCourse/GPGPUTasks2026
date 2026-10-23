# Task05: NVIDIA CUB library baseline

This implementation calls **`cub::DeviceRadixSort::SortKeys` from NVIDIA CUB**,
provided by the installed CUDA Toolkit. It is a reference/library baseline,
**not an independently implemented radix sort or custom Onesweep**. CUB selects
its own architecture-specific implementation; performance depends on the GPU
and the installed CUB/CUDA version.

## Implementation and timing

- One selected CUDA device and the course framework's stream; no multi-GPU work.
- Unsigned 32-bit ascending sort over **all bits `[0, 32)`**, independent of the
  input distribution. The wrapper accepts `0 <= n <= INT_MAX` keys.
- CUB's separate-input/output pointer API leaves the input unchanged. Input,
  output and temporary storage must be disjoint allocations.
- Temporary storage is queried and allocated **once before timing**, then reused.
  No host transfers or allocations in the timed loop.
- One complete synchronized warm-up, followed by the original **10 measured
  sorts of 100,000,000 keys**. Every invocation sorts the original unsorted input.
- Wall-clock timing includes the complete sort and `cudaStreamSynchronize`,
  not just kernel launch overhead. No forced synchronization between CUB passes.
- The benchmark retains the original random input, `std::sort` reference, output
  comparison and input-preservation check. Its input uses values up to `INT_MAX`;
  the separate correctness tests also exercise the full unsigned 32-bit range.
- The reported effective bandwidth is the template's idealized `8*n/time`,
  **not** the actual traffic across all internal radix-sort passes.

## Build and run

Install the original repository prerequisites (including Vulkan/OpenCL build
headers), plus the CUDA Toolkit. No external CUB download is needed.

```sh
# Example for RTX 3090. Choose the architecture matching your GPU/toolkit.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGPU_CUDA_SUPPORT=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build build --parallel 6
./build/test_cub_radix_sort 0
./build/main_radix_sort 0
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/test_cub_radix_sort 0
```

The argument is the **framework's device index**, printed at startup, not
necessarily the `nvidia-smi` index. All sorting runs on that single device.
For V100, use an appropriate toolkit that still supports Volta (e.g. CUDA 11.8
with `-DCMAKE_CUDA_ARCHITECTURES=70`); CUDA 13 removed offline compilation for
Volta. A successful `sm_70` build is not a V100 performance measurement.

`test_cub_radix_sort` checks 128 size/distribution combinations against
`std::sort`: empty and singleton arrays, warp/block boundaries, nonmultiple
sizes up to 1,000,003, random full-range keys, zero/max-value arrays, duplicates,
values around the sign bit, and sorted/reversed input. Allocations are reused
for unrelated inputs; the output is poisoned before each sort. Every nonempty
case checks the entire output and verifies that the input is unchanged.

With CUDA disabled the project still builds, but this CUDA-only benchmark skips
unsupported devices; it does not claim to implement OpenCL or CPU sorting.
