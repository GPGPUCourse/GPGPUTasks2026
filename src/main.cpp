#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


std::string deviceTypeName(cl_device_type deviceType) {
    static std::pair<cl_device_type, std::string> const flags[] = {
        {CL_DEVICE_TYPE_CPU, "CPU"},
        {CL_DEVICE_TYPE_GPU, "GPU"},
        {CL_DEVICE_TYPE_ACCELERATOR, "accelerator"},
        {CL_DEVICE_TYPE_DEFAULT, "default"},
        {CL_DEVICE_TYPE_CUSTOM, "custom"},
    };
    std::string s;
    for (auto const& [flag, name] : flags) {
        if (deviceType & flag) {
            if (!s.empty()) {
                s += "-";
            }
            s += name;
        }
    }
    return s;
}

void reportError(cl_int err, std::string const& filename, int line)
{
    if(CL_SUCCESS == err) {
        return;
    }

    std::string message = "OpenCL error code " + std::to_string(err) + " encountered at " + filename + ":" + std::to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

template<typename T, typename QueryFunc, typename Handle>
T getInfo(QueryFunc query, Handle handle, cl_uint name)
{
    if constexpr (std::is_same_v<T, std::string>) {
        size_t size = 0;
        OCL_SAFE_CALL(query(handle, name, 0, nullptr, &size));
        if (size == 0) {
            throw std::runtime_error("Zero-size field requested: " + std::to_string(name));
        }
        std::string value(size - 1, '\0');
        OCL_SAFE_CALL(query(handle, name, size, value.data(), nullptr));
        return value;
    } else {
        T value{};
        OCL_SAFE_CALL(query(handle, name, sizeof(T), &value, nullptr));
        return value;
    }
}

template<typename T>
T getPlatformInfo(cl_platform_id platform, cl_platform_info name)
{
    return getInfo<T>(clGetPlatformInfo, platform, name);
}

template<typename T>
T getDeviceInfo(cl_device_id device, cl_device_info name)
{
    return getInfo<T>(clGetDeviceInfo, device, name);
}

void printOpenCLInfo() {
    if(!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }

    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for(cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
    {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // 1.2
        std::cout << "\tPlatform name: " << getPlatformInfo<std::string>(platform, CL_PLATFORM_NAME) << std::endl;

        // 1.3
        std::cout << "\tPlatform vendor: " << getPlatformInfo<std::string>(platform, CL_PLATFORM_VENDOR) << std::endl;

        // 2.1
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "\tNumber of devices: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for(cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
        {
            // 2.2
            std::cout << "\tDevice #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            std::cout << "\t\tDevice name: " << getDeviceInfo<std::string>(device, CL_DEVICE_NAME) << std::endl;
            std::cout << "\t\tDevice type: " << deviceTypeName(getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;

            cl_ulong globalMem = getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            std::cout << "\t\tDevice global mem size: " << (globalMem / 1024 / 1024) << " MB" << std::endl;

            std::cout << "\t\tDevice global mem cache size: " << getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) << " B" << std::endl;
            std::cout << "\t\tDevice global mem cacheline size: " << getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << " B" << std::endl;
            cl_ulong localMem = getDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
            std::cout << "\t\tDevice local mem size: " << (localMem / 1024) << " KB" << std::endl;

            std::cout << "\t\tDevice clock frequency: " << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;
            std::cout << "\t\tDevice compute units: " << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
            std::cout << "\t\tDevice work group size: " << getDeviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE) << std::endl;

            std::cout << "\t\tDevice extensions: " << getDeviceInfo<std::string>(device, CL_DEVICE_EXTENSIONS) << std::endl;
        }
    }
}

int main()
{
    try {
        printOpenCLInfo();
    } catch (std::exception const& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
