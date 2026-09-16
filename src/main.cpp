#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include "traits.hpp"

#include <cstring>
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

namespace {

std::vector<unsigned char> fetchRawParam(cl_device_id id, cl_device_info param_name)
{
	size_t param_size = 0;
	OCL_SAFE_CALL(clGetDeviceInfo(id, param_name, 0, nullptr, &param_size));

	std::vector<unsigned char> ret(param_size);
	OCL_SAFE_CALL(clGetDeviceInfo(id, param_name, param_size, ret.data(), nullptr));
	return ret;
}

std::string mapDeviceType(cl_device_type type)
{
	// custom is defined as not being one of the combined types
	if(type == CL_DEVICE_TYPE_CUSTOM)
	{
		return "custom";
	}

	std::ostringstream ss;
	if(type & CL_DEVICE_TYPE_DEFAULT)
	{
		ss << "default, ";
	}

	if(type & CL_DEVICE_TYPE_CPU)
	{
		ss << "cpu, ";
	}

	if(type & CL_DEVICE_TYPE_GPU)
	{
		ss << "gpu, ";
	}

	if(type & CL_DEVICE_TYPE_ACCELERATOR)
	{
		ss << "accelerator, ";
	}

	auto string = ss.str();
	string.resize(string.size() - 2);
	return string;
}

}  // namespace

template<cl_device_info info>
auto fetchParam(cl_device_id id) -> std::enable_if_t<std::is_trivially_copyable_v<cl::param_traits_t<info>>, cl::param_traits_t<info>>
{
	auto raw = fetchRawParam(id, info);
	cl::param_traits_t<info> t;
	std::memcpy(&t, raw.data(), raw.size());
	return t;
}

template<cl_device_info info>
auto fetchParam(cl_device_id id) -> std::enable_if_t<std::is_same_v<cl::param_traits_t<info>, std::string>, cl::param_traits_t<info>>
{
	auto raw = fetchRawParam(id, info);
	const char *cdata = reinterpret_cast<const char *>(raw.data());
	std::string res(cdata, raw.size());
	return res;
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
		std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> deviceIds(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, deviceIds.data(), nullptr));

		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			auto device = deviceIds[deviceIndex];

			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
			std::cout << "      Device name: " << fetchParam<CL_DEVICE_NAME>(device) << std::endl;
			std::cout << "      Device type: " << mapDeviceType(fetchParam<CL_DEVICE_TYPE>(device)) << std::endl;
			std::cout << "      Memory: " << fetchParam<CL_DEVICE_GLOBAL_MEM_SIZE>(device) / (1048576) << " MB" << std::endl;
			std::cout << "      Has unified memory: " << (fetchParam<CL_DEVICE_HOST_UNIFIED_MEMORY>(device) ? "yes" : "no") << std::endl;
			std::cout << "      Max clock frequency: " << fetchParam<CL_DEVICE_MAX_CLOCK_FREQUENCY>(device) << " MHZ" << std::endl;
			std::cout << "      Max compute units: " << fetchParam<CL_DEVICE_MAX_COMPUTE_UNITS>(device) << std::endl;
		}
	}

	return 0;
}
