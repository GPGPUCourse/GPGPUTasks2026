//! Задание 1 (task01) — Rust-зеркало: сложение матриц, coalesced memory access.
//! Тот же эксперимент, что в C++ `src/main_aplusb_matrix.cpp`:
//! GOOD-кернел (coalesced) vs BAD-кернел (stride = ширина строки).
//! Host вместо libgpu — крейт opencl3.

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_ALL};
use opencl3::kernel::Kernel;
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use std::ptr;
use std::time::Instant;

// Группа 16x16 = 256 work-items, как GROUP_SIZE_X/GROUP_SIZE_Y в kernels/defines.h
const LOCAL_X: usize = 16;
const LOCAL_Y: usize = 16;

const KERNELS: &str = r#"
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void aplusb_matrix_good(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    // ХОРОШИЙ вариант: соседние work-items по измерению 0 читают/пишут
    // соседние элементы одной строки -> coalesced доступ.
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    const unsigned int index = y * width + x;
    c[index] = a[index] + b[index];
}

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    // ПЛОХОЙ вариант: запуск на "перевернутом" пространстве height x width.
    // Соседние work-items по измерению 0 отличаются на целую строку
    // (stride = width элементов) -> каждая транзакция почти впустую.
    // Каждый элемент считается ровно один раз, результат корректен.
    const unsigned int row = get_global_id(0);
    const unsigned int col = get_global_id(1);
    if (col >= width || row >= height)
        return;
    const unsigned int index = row * width + col;
    c[index] = a[index] + b[index];
}
"#;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn main() -> Result<(), String> {
    let task_size: u32 = 64;
    let width: u32 = task_size * 256; // 16384
    let height: u32 = task_size * 128; // 8192
    let n = (width as usize) * (height as usize);
    println!(
        "matrices size: {}x{} = 3 * {} MB",
        width,
        height,
        std::mem::size_of::<u32>() * n / 1024 / 1024
    );

    let platforms = get_platforms().map_err(|e| format!("get_platforms: {e}"))?;
    if platforms.is_empty() {
        return Err("no OpenCL platforms".to_string());
    }
    let platform = &platforms[0];
    println!(
        "Platform: {} ({})",
        platform.name().unwrap_or_default(),
        platform.vendor().unwrap_or_default()
    );
    let device_ids = platform
        .get_devices(CL_DEVICE_TYPE_ALL)
        .map_err(|e| format!("get_devices: {e}"))?;
    if device_ids.is_empty() {
        return Err("no OpenCL devices".to_string());
    }
    let device = Device::new(device_ids[0]);
    println!("Device: {}", device.name().unwrap_or_default());

    let context = Context::from_device(&device).map_err(|e| format!("context: {e}"))?;
    let queue = CommandQueue::create_default(&context, 0).map_err(|e| format!("queue: {e}"))?;
    let program = Program::create_and_build_from_source(&context, KERNELS, "")
        .map_err(|e| format!("build program: {e}"))?;
    let good =
        Kernel::create(&program, "aplusb_matrix_good").map_err(|e| format!("kernel good: {e}"))?;
    let bad =
        Kernel::create(&program, "aplusb_matrix_bad").map_err(|e| format!("kernel bad: {e}"))?;

    let mut as_vec = vec![0u32; n];
    let mut bs_vec = vec![0u32; n];
    for (i, (a, b)) in as_vec.iter_mut().zip(bs_vec.iter_mut()).enumerate() {
        *a = 3 * (i as u32 + 5) + 7;
        *b = 11 * (i as u32 + 13) + 17;
    }

    let mut a_buf: Buffer<u32>;
    let mut b_buf: Buffer<u32>;
    let c_buf: Buffer<u32>;
    unsafe {
        a_buf = Buffer::create(&context, CL_MEM_READ_WRITE, n, ptr::null_mut())
            .map_err(|e| format!("buffer a: {e}"))?;
        b_buf = Buffer::create(&context, CL_MEM_READ_WRITE, n, ptr::null_mut())
            .map_err(|e| format!("buffer b: {e}"))?;
        c_buf = Buffer::create(&context, CL_MEM_READ_WRITE, n, ptr::null_mut())
            .map_err(|e| format!("buffer c: {e}"))?;
        queue
            .enqueue_write_buffer(&mut a_buf, 1, 0, &as_vec, &[])
            .map_err(|e| format!("write a: {e}"))?;
        queue
            .enqueue_write_buffer(&mut b_buf, 1, 0, &bs_vec, &[])
            .map_err(|e| format!("write b: {e}"))?;
    }

    // За один запуск через память проходит 3 матрицы
    let traffic_gb = 3.0 * std::mem::size_of::<u32>() as f64 * n as f64 / 1024.0 / 1024.0 / 1024.0;

    // ---------- BAD ----------
    println!("Running BAD matrix kernel...");
    unsafe {
        bad.set_arg(0, &a_buf).map_err(|e| format!("arg: {e}"))?;
        bad.set_arg(1, &b_buf).map_err(|e| format!("arg: {e}"))?;
        bad.set_arg(2, &c_buf).map_err(|e| format!("arg: {e}"))?;
        bad.set_arg(3, &width).map_err(|e| format!("arg: {e}"))?;
        bad.set_arg(4, &height).map_err(|e| format!("arg: {e}"))?;
    }
    // "Перевернутое" пространство: dim0 = height, dim1 = width
    let bad_global = [height as usize, width as usize];
    let local = [LOCAL_X, LOCAL_Y];
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let t = Instant::now();
        unsafe {
            queue
                .enqueue_nd_range_kernel(
                    bad.get(),
                    2,
                    ptr::null(),
                    bad_global.as_ptr(),
                    local.as_ptr(),
                    &[],
                )
                .map_err(|e| format!("enqueue bad: {e}"))?;
        }
        queue.finish().map_err(|e| format!("finish: {e}"))?;
        times.push(t.elapsed().as_secs_f64());
    }
    println!(
        "BAD times: min={:.5} median={:.5} max={:.5} s",
        times.iter().cloned().fold(f64::INFINITY, f64::min),
        median(times.clone()),
        times.iter().cloned().fold(0.0, f64::max)
    );
    println!(
        "BAD median VRAM bandwidth: {:.4} GB/s",
        traffic_gb / median(times)
    );
    let mut cs_vec = vec![0u32; n];
    unsafe {
        queue
            .enqueue_read_buffer(&c_buf, 1, 0, &mut cs_vec, &[])
            .map_err(|e| format!("read c: {e}"))?;
    }
    for i in 0..n {
        assert_eq!(
            cs_vec[i],
            as_vec[i].wrapping_add(bs_vec[i]),
            "BAD mismatch at {i}"
        );
    }
    println!("BAD result verified");

    // ---------- GOOD ----------
    println!("Running GOOD matrix kernel...");
    unsafe {
        good.set_arg(0, &a_buf).map_err(|e| format!("arg: {e}"))?;
        good.set_arg(1, &b_buf).map_err(|e| format!("arg: {e}"))?;
        good.set_arg(2, &c_buf).map_err(|e| format!("arg: {e}"))?;
        good.set_arg(3, &width).map_err(|e| format!("arg: {e}"))?;
        good.set_arg(4, &height).map_err(|e| format!("arg: {e}"))?;
    }
    let good_global = [width as usize, height as usize];
    let mut times = Vec::with_capacity(10);
    for _ in 0..10 {
        let t = Instant::now();
        unsafe {
            queue
                .enqueue_nd_range_kernel(
                    good.get(),
                    2,
                    ptr::null(),
                    good_global.as_ptr(),
                    local.as_ptr(),
                    &[],
                )
                .map_err(|e| format!("enqueue good: {e}"))?;
        }
        queue.finish().map_err(|e| format!("finish: {e}"))?;
        times.push(t.elapsed().as_secs_f64());
    }
    println!(
        "GOOD times: min={:.5} median={:.5} max={:.5} s",
        times.iter().cloned().fold(f64::INFINITY, f64::min),
        median(times.clone()),
        times.iter().cloned().fold(0.0, f64::max)
    );
    println!(
        "GOOD median VRAM bandwidth: {:.4} GB/s",
        traffic_gb / median(times)
    );
    unsafe {
        queue
            .enqueue_read_buffer(&c_buf, 1, 0, &mut cs_vec, &[])
            .map_err(|e| format!("read c: {e}"))?;
    }
    for i in 0..n {
        assert_eq!(
            cs_vec[i],
            as_vec[i].wrapping_add(bs_vec[i]),
            "GOOD mismatch at {i}"
        );
    }
    println!("GOOD result verified");
    Ok(())
}
