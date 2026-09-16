#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::string printable_cl_device_type(cl_device_type type)
{
	static constexpr std::pair<cl_device_type, const char *> names[] = {
		{ CL_DEVICE_TYPE_DEFAULT, "DEFAULT" },
		{ CL_DEVICE_TYPE_CPU, "CPU" },
		{ CL_DEVICE_TYPE_GPU, "GPU" },
		{ CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR" },
#ifdef CL_VERSION_1_2
		{ CL_DEVICE_TYPE_CUSTOM, "CUSTOM" },
#endif
	};

	std::string result;
	for(const auto &[bit, name] : names)
	{
		if(type & bit)
		{
			if(!result.empty())
			{
				result += '|';
			}
			result += name;
		}
	}

	return result.empty() ? "UNKNOWN" : result;
}

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
	{
		return;
	}

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

inline cl_int queryInfo(cl_platform_id handle, cl_uint param, size_t size, void *value, size_t *sizeRet)
{
	return clGetPlatformInfo(handle, param, size, value, sizeRet);
}

inline cl_int queryInfo(cl_device_id handle, cl_uint param, size_t size, void *value, size_t *sizeRet)
{
	return clGetDeviceInfo(handle, param, size, value, sizeRet);
}

template<typename T, typename Handle>
T getInfo(Handle handle, cl_uint param)
{
	T value{};
	OCL_SAFE_CALL(queryInfo(handle, param, sizeof(T), static_cast<void *>(&value), nullptr));
	return value;
}

template<typename Handle>
std::string getInfoStr(Handle handle, cl_uint param)
{
	std::size_t size = 0;
	OCL_SAFE_CALL(queryInfo(handle, param, 0, nullptr, &size));

	std::string value(size, '\0');
	OCL_SAFE_CALL(queryInfo(handle, param, size, static_cast<void *>(value.data()), nullptr));
	if(!value.empty() && value.back() == '\0')
	{
		value.pop_back();
	}

	return value;
}

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
	{
		throw std::runtime_error("Can't init OpenCL driver!");
	}

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	if(platformsCount > 0)
	{
		OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
	}

	for(cl_uint platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		// TODO 1.1
		// std::size_t platformNameSize = 0;
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
		/*
		 * Number of OpenCL platforms: 2
		 * Platform #1/2
		 * terminate called after throwing an instance of 'std::runtime_error'
		 *   what():  OpenCL error code -30 encountered at /home/shard/VsCode/GPGPUTasks2026/src/main.cpp:128
		 * fish: Job 1, './enumDevices' terminated by signal SIGABRT (Abort)
		 *
		 * #define CL_INVALID_VALUE                            -30 <---
		 * param_name is not one of the supported values
		 */

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::cout << "    Platform name: " << getInfoStr(platform, CL_PLATFORM_NAME) << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
		std::cout << "    Vendor name: " << getInfoStr(platform, CL_PLATFORM_VENDOR) << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

		std::vector<cl_device_id> devices(devicesCount);
		if(devicesCount > 0)
		{
			OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		}

		for(cl_uint deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			const cl_device_id device_id = devices[deviceIndex];
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			/* Device name */
			std::cout << "		Device name: " << getInfoStr(device_id, CL_DEVICE_NAME) << std::endl;

			/* Device type */
			{
				const auto deviceType = getInfo<cl_device_type>(device_id, CL_DEVICE_TYPE);
				std::cout << "		Device type: " << printable_cl_device_type(deviceType) << std::endl;
			}

			/* Memory size, MB */
			{
				const auto memorySizeBytes = getInfo<cl_ulong>(device_id, CL_DEVICE_GLOBAL_MEM_SIZE);
				std::cout << "		Memory size (MB): " << (memorySizeBytes / 1024 / 1024) << std::endl;
			}

			/* Extensions */
			std::cout << "		Extensions: " << getInfoStr(device_id, CL_DEVICE_EXTENSIONS) << std::endl;

			/* Vendor ID */
			{
				const auto vendor_id = getInfo<cl_uint>(device_id, CL_DEVICE_VENDOR_ID);
				std::cout << "		Vendor ID: 0x" << std::hex << vendor_id << std::dec << std::endl;
			}
		}
	}

	return 0;
}
