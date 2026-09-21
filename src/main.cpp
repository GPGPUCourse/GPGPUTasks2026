#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iomanip>
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

template<typename T>
T getDeviceInfo(cl_device_id device, cl_device_info param)
{
	T value{};
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, sizeof(T), &value, nullptr));
	return value;
}

template<typename T>
std::vector<T> getDeviceInfoArray(cl_device_id device, cl_device_info param)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, 0, nullptr, &size));
	std::vector<T> value(size / sizeof(T));
	OCL_SAFE_CALL(clGetDeviceInfo(device, param, size, value.data(), nullptr));
	return value;
}

template<typename T>
std::vector<T> getPlatformInfoArray(cl_platform_id platform, cl_platform_info param)
{
	size_t size = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, 0, nullptr, &size));
	std::vector<T> value(size / sizeof(T));
	OCL_SAFE_CALL(clGetPlatformInfo(platform, param, size, value.data(), nullptr));
	return value;
}

static constexpr const char *kRootIndent = "";
static constexpr const char *kPlatformIndent = "    ";
static constexpr const char *kDeviceIndent = "      ";

std::ostream &header(const char *what, cl_uint index, cl_uint total, const char *indent)
{
	return std::cout << indent << what << " #" << (index + 1) << "/" << total;
}

std::ostream &field(const char *label, const char *indent)
{
	return std::cout << indent << std::left << std::setw(24) << label;
}

std::string deviceTypeToString(cl_device_type type)
{
	static constexpr std::pair<cl_device_type, const char *> kTypes[] = {
		{ CL_DEVICE_TYPE_CPU, "CPU" },
		{ CL_DEVICE_TYPE_GPU, "GPU" },
		{ CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR" },
		{ CL_DEVICE_TYPE_CUSTOM, "CUSTOM" },
		{ CL_DEVICE_TYPE_DEFAULT, "DEFAULT" },
	};

	std::string result;
	for(auto [bit, name] : kTypes)
		if(type & bit)
			result += (result.empty() ? "" : " | ") + std::string(name);

	return result.empty() ? "UNKNOWN" : result;
}

int main()
{
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(cl_int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		cl_platform_id platform = platforms[platformIndex];

		header("Platform", platformIndex, platformsCount, kRootIndent) << std::endl;
		field("Platform name:", kPlatformIndent) << getPlatformInfoArray<unsigned char>(platform, CL_PLATFORM_NAME).data() << std::endl;
		field("Vendor name:", kPlatformIndent) << getPlatformInfoArray<unsigned char>(platform, CL_PLATFORM_VENDOR).data() << std::endl;

		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::vector<cl_device_id> devices(devicesCount, 0);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(cl_int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			cl_device_id device = devices[deviceIndex];

			header("Device", deviceIndex, devicesCount, kPlatformIndent) << std::endl;
			field("Device name:", kDeviceIndent) << getDeviceInfoArray<unsigned char>(device, CL_DEVICE_NAME).data() << std::endl;
			field("Device type:", kDeviceIndent) << deviceTypeToString(getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;
			field("Memory:", kDeviceIndent) << (getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE) >> 20) << " MiB" << std::endl;
			field("Max compute units:", kDeviceIndent) << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
			field("Max clock frequency:", kDeviceIndent) << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;
			field("Max work group size:", kDeviceIndent) << getDeviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE) << std::endl;
			field("Driver version:", kDeviceIndent) << getDeviceInfoArray<unsigned char>(device, CL_DRIVER_VERSION).data() << std::endl;
		}
	}

	return 0;
}
