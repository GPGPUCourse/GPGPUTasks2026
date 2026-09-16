#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
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

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

namespace cl_utils {
template<typename, template<typename...> class T>
inline constexpr bool is_specialization_of_v = std::false_type{};

template<template<typename...> typename T, typename... Args>
inline constexpr bool is_specialization_of_v<T<Args...>, T> = std::true_type{};

template<typename T>
auto getDeviceInfo(cl_device_id device, cl_device_info paramName)
{
	size_t paramValueSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, 0, nullptr, &paramValueSize));
	if constexpr(is_specialization_of_v<T, std::vector> || std::is_same_v<T, std::string>)
	{
		if(paramValueSize % sizeof(typename T::value_type))
			throw std::runtime_error("Unexpected size of device info parameter value");
		size_t numValues = paramValueSize / sizeof(typename T::value_type);
		if constexpr(std::is_same_v<T, std::string>)
			--numValues;  // Skip null terminator
		auto result = T(numValues, 0);
		OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, paramValueSize, std::data(result), nullptr));
		return result;
	}
	else
	{
		if(paramValueSize != sizeof(T))
			throw std::runtime_error("Unexpected size of device info parameter value");
		T result;
		OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, paramValueSize, std::addressof(result), nullptr));
		return result;
	}
}

std::string deviceTypeToString(const cl_device_type &type)
{
	if(type == CL_DEVICE_TYPE_CUSTOM)
		return "Custom";
	std::stringstream ss;
	if(type & CL_DEVICE_TYPE_CPU)
		ss << "CPU" << '+';
	if(type & CL_DEVICE_TYPE_GPU)
		ss << "GPU" << '+';
	if(type & CL_DEVICE_TYPE_ACCELERATOR)
		ss << "Accelerator" << '+';
	if(type & CL_DEVICE_TYPE_DEFAULT)
		ss << "Default" << '+';
	std::string result = ss.str();
	if(!result.empty()) result.pop_back();
	return result;
}

}  // namespace cl_utils

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;
		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			std::cout << "        Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			cl_device_id device = devices[deviceIndex];

			/*
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device,  CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "        Device name: " << deviceName.data() << std::endl;
			*/

			auto deviceVendor = cl_utils::getDeviceInfo<std::string>(device, CL_DEVICE_VENDOR);
			auto deviceVendorId = cl_utils::getDeviceInfo<cl_uint>(device, CL_DEVICE_VENDOR_ID);
			std::stringstream vendorIdSs;
			vendorIdSs << std::setw(4) << std::setfill('0') << std::hex << std::uppercase << deviceVendorId;
			std::cout << "        Device vendor: " << deviceVendor << " (ID: " << vendorIdSs.str() << ")" << std::endl;

			auto deviceName = cl_utils::getDeviceInfo<std::string>(device, CL_DEVICE_NAME);
			std::cout << "        Device name: " << deviceName << std::endl;

			auto deviceType = cl_utils::getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE);
			std::cout << "        Device type: " << cl_utils::deviceTypeToString(deviceType) << std::endl;

			auto deviceVersion = cl_utils::getDeviceInfo<std::string>(device, CL_DEVICE_VERSION);
			std::cout << "        Device version: " << deviceVersion << std::endl;

			auto deviceOpenclCVersion = cl_utils::getDeviceInfo<std::string>(device, CL_DEVICE_OPENCL_C_VERSION);
			std::cout << "        OpenCL C version: " << deviceOpenclCVersion << std::endl;

			auto deviceDriverVersion = cl_utils::getDeviceInfo<std::string>(device, CL_DRIVER_VERSION);
			std::cout << "        Driver version: " << deviceDriverVersion << std::endl;

			auto deviceProfile = cl_utils::getDeviceInfo<std::string>(device, CL_DEVICE_PROFILE);
			std::cout << "        Device profile: " << deviceProfile << std::endl;

			auto deviceGlobalMemSize = cl_utils::getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "        Global memory size: " << deviceGlobalMemSize / (1024 * 1024) << " MB" << std::endl;

			auto deviceLocalMemSize = cl_utils::getDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
			std::cout << "        Local memory size: " << deviceLocalMemSize / (1024) << " KB" << std::endl;

			auto deviceHostUnifiedMemory = cl_utils::getDeviceInfo<cl_bool>(device, CL_DEVICE_HOST_UNIFIED_MEMORY);
			std::cout << "        Host unified memory: " << (deviceHostUnifiedMemory ? "Yes" : "No") << std::endl;

			auto deviceMaxComputeUnits = cl_utils::getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
			std::cout << "        Max compute units: " << deviceMaxComputeUnits << std::endl;

			auto deviceMaxWorkGroupSize = cl_utils::getDeviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
			std::cout << "        Max work group size: " << deviceMaxWorkGroupSize << std::endl;

			auto deviceMaxWorkItemSizes = cl_utils::getDeviceInfo<std::vector<size_t>>(device, CL_DEVICE_MAX_WORK_ITEM_SIZES);
			std::cout << "        Max work item sizes: ";
			for(size_t i = 0; i < deviceMaxWorkItemSizes.size(); ++i)
			{
				if(i > 0) std::cout << " x ";
				std::cout << deviceMaxWorkItemSizes[i];
			}
			std::cout << std::endl;
		}
	}

	return 0;
}
