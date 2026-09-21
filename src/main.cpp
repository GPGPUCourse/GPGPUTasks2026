#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::string getPlatformString(cl_platform_id platform, cl_platform_info parameter)
{
	size_t valueSize = 0;
	OCL_SAFE_CALL(clGetPlatformInfo(platform, parameter, 0, nullptr, &valueSize));

	std::vector<char> value(valueSize, '\0');
	OCL_SAFE_CALL(clGetPlatformInfo(platform, parameter, value.size(), value.data(), nullptr));
	return value.data();
}

std::string getDeviceString(cl_device_id device, cl_device_info parameter)
{
	size_t valueSize = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(device, parameter, 0, nullptr, &valueSize));

	std::vector<char> value(valueSize, '\0');
	OCL_SAFE_CALL(clGetDeviceInfo(device, parameter, value.size(), value.data(), nullptr));
	return value.data();
}

template<typename T>
T getDeviceValue(cl_device_id device, cl_device_info parameter)
{
	T value{};
	OCL_SAFE_CALL(clGetDeviceInfo(device, parameter, sizeof(value), &value, nullptr));
	return value;
}

std::string getDeviceTypeName(cl_device_type type)
{
	std::vector<std::string> names;
	if(type & CL_DEVICE_TYPE_DEFAULT)
		names.emplace_back("default");
	if(type & CL_DEVICE_TYPE_CPU)
		names.emplace_back("CPU");
	if(type & CL_DEVICE_TYPE_GPU)
		names.emplace_back("GPU");
	if(type & CL_DEVICE_TYPE_ACCELERATOR)
		names.emplace_back("accelerator");
	if(type & CL_DEVICE_TYPE_CUSTOM)
		names.emplace_back("custom");

	if(names.empty())
		return "unknown";

	std::ostringstream result;
	for(size_t i = 0; i < names.size(); ++i)
	{
		if(i != 0)
			result << " | ";
		result << names[i];
	}
	return result.str();
}

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

	for(cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
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

		// Для некорректного param_name метод возвращает CL_INVALID_VALUE (-30).
		std::cout << "    Platform name: " << getPlatformString(platform, CL_PLATFORM_NAME) << std::endl;
		std::cout << "    Platform vendor: " << getPlatformString(platform, CL_PLATFORM_VENDOR) << std::endl;

		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			cl_device_id device = devices[deviceIndex];
			const cl_device_type deviceType = getDeviceValue<cl_device_type>(device, CL_DEVICE_TYPE);
			const cl_ulong globalMemory = getDeviceValue<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			const cl_uint computeUnits = getDeviceValue<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
			const cl_uint clockFrequency = getDeviceValue<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
			const size_t maxWorkGroupSize = getDeviceValue<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);

			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			std::cout << "        Name: " << getDeviceString(device, CL_DEVICE_NAME) << std::endl;
			std::cout << "        Vendor: " << getDeviceString(device, CL_DEVICE_VENDOR) << std::endl;
			std::cout << "        Type: " << getDeviceTypeName(deviceType) << std::endl;
			std::cout << "        Global memory: " << (globalMemory / (1024 * 1024)) << " MB" << std::endl;
			std::cout << "        Compute units: " << computeUnits << std::endl;
			std::cout << "        Max clock frequency: " << clockFrequency << " MHz" << std::endl;
			std::cout << "        Max work-group size: " << maxWorkGroupSize << std::endl;
			std::cout << "        OpenCL version: " << getDeviceString(device, CL_DEVICE_VERSION) << std::endl;
		}
	}

	return 0;
}
