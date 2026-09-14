#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value)
{
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		if (platformIndex == 0)
		{
			try
			{
				OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
			} catch (std::runtime_error& e)
			{
				std::cout << "  Error test: " << e.what() << std::endl;
			}
		}
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendor.size(), platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceName.size(), deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;

			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			std::cout << "        Device type: ";
			if (deviceType & CL_DEVICE_TYPE_GPU)         std::cout << "GPU ";
			if (deviceType & CL_DEVICE_TYPE_CPU)         std::cout << "CPU ";
			if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "Accelerator ";
			if (deviceType & CL_DEVICE_TYPE_DEFAULT)	 std::cout << "Default ";
			if (deviceType & CL_DEVICE_TYPE_CUSTOM)      std::cout << "Custom ";
			std::cout << std::endl;

			cl_ulong deviceGlobalMemSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceGlobalMemSize), &deviceGlobalMemSize, nullptr));
			std::cout << "        Global memory size: " << (deviceGlobalMemSize / (1024 * 1024)) << " MB" << std::endl;

			cl_bool isDeviceLittleEndian = CL_FALSE;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(isDeviceLittleEndian), &isDeviceLittleEndian, nullptr));
			std::cout << "        Endian little: " << (isDeviceLittleEndian ? "true" : "false") << std::endl;

			cl_ulong deviceGlobalMemCacheSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceGlobalMemCacheSize), &deviceGlobalMemCacheSize, nullptr));
			std::cout << "        Global memory cache size: " << (deviceGlobalMemCacheSize / 1024) << " KB" << std::endl;

			cl_ulong deviceMaxMemAllocSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(deviceMaxMemAllocSize), &deviceMaxMemAllocSize, nullptr));
			std::cout << "        Max memory allocation: " << (deviceMaxMemAllocSize / (1024 * 1024)) << " MB" << std::endl;

			cl_platform_id devicePlatform = nullptr;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(devicePlatform), &devicePlatform, nullptr));
			std::cout << "        Belongs to same platform: " << (devicePlatform == platform ? "true" : "false") << std::endl;
		}
	}

	return 0;
}
