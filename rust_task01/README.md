# Task01 — Rust-зеркало (сложение матриц, coalesced access)

Тот же эксперимент, что в C++ `src/main_aplusb_matrix.cpp`:
GOOD-кернел (coalesced) vs BAD-кернел (stride = ширина строки).
Host вместо `libgpu` — крейт `opencl3` напрямую, кернельная логика та же.

Зависимость: `opencl3 = "0.12"`.

## Проверка в WSL (Ubuntu 24.04, pocl CPU)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --release
./target/release/rust_task01
```

Замер (матрицы 16384x8192, 3 x 512 MB):

```text
BAD median VRAM bandwidth: ~1.65 GB/s
GOOD median VRAM bandwidth: ~12.8 GB/s
```

Разница ~8x — совпадает с C++-замером (1.38 vs 12.53 GB/s).

## Важно для сдачи

Штатный CI курса собирает C++ (`main_aplusb_matrix`).
Rust-код лежит отдельной папкой и не трогает `CMakeLists.txt`/CI.
