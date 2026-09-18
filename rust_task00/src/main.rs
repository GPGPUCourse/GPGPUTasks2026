//! Задание 0 (task00) — порт на Rust.
//! Перечисление OpenCL платформ и устройств.
//! Соответствует TODO 1.1, 1.2, 1.3, 2.1, 2.2 из src/main.cpp.

use opencl3::device::{
    Device, CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU,
    CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_GPU,
};
use opencl3::platform::get_platforms;

fn device_type_to_string(dtype: u64) -> String {
    // dtype — bitfield, может содержать несколько битов
    let mut parts = Vec::new();
    if dtype & CL_DEVICE_TYPE_CPU != 0 {
        parts.push("CPU");
    }
    if dtype & CL_DEVICE_TYPE_GPU != 0 {
        parts.push("GPU");
    }
    if dtype & CL_DEVICE_TYPE_ACCELERATOR != 0 {
        parts.push("ACCELERATOR");
    }
    if dtype & CL_DEVICE_TYPE_CUSTOM != 0 {
        parts.push("CUSTOM");
    }
    if dtype & CL_DEVICE_TYPE_DEFAULT != 0 {
        parts.push("DEFAULT");
    }
    if parts.is_empty() {
        format!("strange device ({dtype})")
    } else {
        parts.join("|")
    }
}

fn main() -> Result<(), String> {
    // TODO 1.1: если запросить clGetPlatformInfo с некорректным param_name
    // (например 239), OpenCL вернёт CL_INVALID_VALUE (-30).
    // В opencl3 это выразилось бы в Err(-30). Здесь проверяем только
    // корректные param_name через безопасные обёртки.

    let platforms = get_platforms().map_err(|e| format!("Can't get OpenCL platforms: {e}"))?;
    println!("Number of OpenCL platforms: {}", platforms.len());

    if platforms.is_empty() {
        return Err("Can't init OpenCL driver! No platforms found.".to_string());
    }

    for (platform_index, platform) in platforms.iter().enumerate() {
        println!("Platform #{}/{}", platform_index + 1, platforms.len());

        // TODO 1.2: имя платформы (CL_PLATFORM_NAME)
        let platform_name = platform
            .name()
            .map_err(|e| format!("clGetPlatformInfo(CL_PLATFORM_NAME) failed: {e}"))?;
        println!("    Platform name: {platform_name}");

        // TODO 1.3: вендор платформы (CL_PLATFORM_VENDOR)
        let platform_vendor = platform
            .vendor()
            .map_err(|e| format!("clGetPlatformInfo(CL_PLATFORM_VENDOR) failed: {e}"))?;
        println!("    Platform vendor: {platform_vendor}");

        // Дополнительно полезно для отладки (не требовалось, но безвредно)
        if let Ok(platform_version) = platform.version() {
            println!("    Platform version: {platform_version}");
        }
        if let Ok(platform_profile) = platform.profile() {
            println!("    Platform profile: {platform_profile}");
        }

        // TODO 2.1: число и список устройств (CL_DEVICE_TYPE_ALL)
        let device_ids = platform
            .get_devices(CL_DEVICE_TYPE_ALL)
            .map_err(|e| format!("clGetDeviceIDs failed: {e}"))?;
        println!("    Number of devices: {}", device_ids.len());

        for (device_index, device_id) in device_ids.iter().enumerate() {
            // TODO 2.2
            let device = Device::new(*device_id);
            println!("    Device #{}/{}", device_index + 1, device_ids.len());

            let device_name = device
                .name()
                .map_err(|e| format!("CL_DEVICE_NAME failed: {e}"))?;
            println!("        Device name: {device_name}");

            let dtype = device
                .dev_type()
                .map_err(|e| format!("CL_DEVICE_TYPE failed: {e}"))?;
            println!("        Device type: {}", device_type_to_string(dtype));

            let global_mem = device
                .global_mem_size()
                .map_err(|e| format!("CL_DEVICE_GLOBAL_MEM_SIZE failed: {e}"))?;
            println!(
                "        Global memory size: {} MB ({global_mem} bytes)",
                global_mem / 1024 / 1024
            );

            // Пара (и больше) интересных свойств
            let max_units = device
                .max_compute_units()
                .map_err(|e| format!("CL_DEVICE_MAX_COMPUTE_UNITS failed: {e}"))?;
            println!("        Max compute units: {max_units}");

            let max_wg = device
                .max_work_group_size()
                .map_err(|e| format!("CL_DEVICE_MAX_WORK_GROUP_SIZE failed: {e}"))?;
            println!("        Max work group size: {max_wg}");

            let max_clock = device
                .max_clock_frequency()
                .map_err(|e| format!("CL_DEVICE_MAX_CLOCK_FREQUENCY failed: {e}"))?;
            println!("        Max clock frequency: {max_clock} MHz");

            let dev_version = device
                .version()
                .map_err(|e| format!("CL_DEVICE_VERSION failed: {e}"))?;
            println!("        Device version: {dev_version}");

            if let Ok(dev_vendor) = device.vendor() {
                println!("        Device vendor: {dev_vendor}");
            }
            if let Ok(local_mem) = device.local_mem_size() {
                println!(
                    "        Local memory size: {} KB ({local_mem} bytes)",
                    local_mem / 1024
                );
            }
        }
    }

    Ok(())
}
