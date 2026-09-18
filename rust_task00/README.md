# Task00 — порт на Rust (enumDevices)

Порт вводного задания (перечисление OpenCL платформ/устройств) на Rust.
Соответствует TODO 1.1, 1.2, 1.3, 2.1, 2.2 из `src/main.cpp` ветки `task00`.

Зависимость: `opencl3 = "0.12"` (безопасные обёртки над `clGetPlatformIDs`,
`clGetPlatformInfo`, `clGetDeviceIDs`, `clGetDeviceInfo`).

## Проверка в WSL (Ubuntu 24.04)

```bash
# один раз: OpenCL CPU-драйвер + тулчейн
sudo apt-get update
sudo apt-get install -y cmake ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo pocl-opencl-icd build-essential pkg-config curl git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# сборка и запуск
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build --release
./target/release/rust_task00
```

Ожидаемый вывод (pocl, CPU):

```text
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: Portable Computing Language
    Platform vendor: The pocl project
    ...
    Number of devices: 1
    Device #1/1
        Device name: ...
        Device type: CPU
        ...
```

## Важно для сдачи

Штатный CI курса собирает C++ (`CMakeLists.txt` → `./enumDevices`).
Чтобы не ломать CI, Rust-код лежит в отдельной папке `rust_task00/`
и не трогает `src/main.cpp` / `CMakeLists.txt`.
В PR приложить вывод и C++ (`./enumDevices`), и Rust (`cargo run`).
