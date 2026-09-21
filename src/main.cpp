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

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

template<typename T, auto Func, typename IdType, typename InfoType>
T getInfo(const IdType id, const InfoType infoType)
{
	size_t InfoSize = 0;
	OCL_SAFE_CALL(Func(id, infoType, 0, nullptr, &InfoSize));
	if constexpr(std::is_same_v<T, std::string>)
	{
		T platformDeviceInfo(InfoSize, 0);
		OCL_SAFE_CALL(Func(id, infoType, InfoSize, platformDeviceInfo.data(), nullptr));
		if (!platformDeviceInfo.empty() && platformDeviceInfo.back() == '\0') {
			platformDeviceInfo.pop_back();
		}
		return platformDeviceInfo;
	}
	else
	{
		T result{};
		OCL_SAFE_CALL(Func(id, infoType, sizeof(T), &result, nullptr));
		return result;
	}
}

template<typename T>
T getPlatformInfo(cl_platform_id id, cl_platform_info param)
{
	return getInfo<T, clGetPlatformInfo>(id, param);
}

template<typename T>
T getDeviceInfo(cl_device_id id, const cl_device_info param)
{
	return getInfo<T, clGetDeviceInfo>(id, param);
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

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		auto platformName = getPlatformInfo<std::string>(platform, CL_PLATFORM_NAME);
		std::cout << "    Platform name: " << platformName << std::endl;

		auto platformVendorName = getPlatformInfo<std::string>(platform, CL_PLATFORM_VENDOR);
		std::cout << "    Platform vendor: " << platformVendorName << std::endl;
		// Запросите и напечатайте так же в консоль вендора данной платформы

		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "Number of OpenCL devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			cl_device_id device = devices[deviceIndex];

			auto deviceName = getDeviceInfo<std::string>(device, CL_DEVICE_NAME);
			std::cout << "    Device name: " << deviceName << std::endl;

			auto deviceType = getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE);
			if(deviceType & CL_DEVICE_TYPE_GPU)
			{
				std::cout << "    Device type: GPU" << std::endl;
			}
			else if(deviceType & CL_DEVICE_TYPE_CPU)
			{
				std::cout << "    Device type: CPU" << std::endl;
			}
			else
			{
				std::cout << "    Device type: something strange" << std::endl;
			}

			auto deviceMemorySize = getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
			std::cout << "    Device global memory size: " << static_cast<double>(deviceMemorySize) / (1024.0 * 1024.0) << " Mb" << std::endl;

			auto deviceProfile = getDeviceInfo<std::string>(device, CL_DEVICE_PROFILE);
			std::cout << "    Device profile: " << deviceProfile << std::endl;

			if(getDeviceInfo<cl_bool>(device, CL_DEVICE_ENDIAN_LITTLE))
			{
				std::cout << "    Little Endian" << std::endl;
			}
			else
			{
				std::cout << "    Big Endian" << std::endl;
			}
		}
	}

	return 0;
}
