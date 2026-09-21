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
	if(CL_SUCCESS == err) return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:178
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


// айм трули сори, если я тут переборщил немного со всеми этими шаблонами, мне просто было немного скучно =)

template<cl_device_info param>
struct DeviceInfoTraits;

template<>
struct DeviceInfoTraits<CL_DEVICE_TYPE>
{
	using type = cl_device_type;
};

template<>
struct DeviceInfoTraits<CL_DEVICE_GLOBAL_MEM_SIZE>
{
	using type = cl_ulong;
};

template<>
struct DeviceInfoTraits<CL_DEVICE_MAX_CLOCK_FREQUENCY>
{
	using type = cl_uint;
};

template<>
struct DeviceInfoTraits<CL_DEVICE_MAX_COMPUTE_UNITS>
{
	using type = cl_uint;
};


template<typename Func, typename Id, typename Param>
std::string getInfoString(Func getInfoFunc, Id id, Param param)
{
	size_t size{0};
	OCL_SAFE_CALL(getInfoFunc(id, param, 0, nullptr, &size));
	std::vector<unsigned char> buffer(size, 0);
	OCL_SAFE_CALL(getInfoFunc(id, param, size, buffer.data(), nullptr));
	return { reinterpret_cast<char *>(buffer.data()) };
}

template<cl_device_info Param, typename Func, typename Id>
typename DeviceInfoTraits<Param>::type getDeviceValue(Func getInfoFunc, Id id)
{
	typename DeviceInfoTraits<Param>::type value{};
	OCL_SAFE_CALL(getInfoFunc(id, Param, sizeof(value), &value, nullptr));
	return value;
}

template<typename T, typename Func, typename... Args>
std::vector<T> getVectorIDs(Func getIDsFunc, Args... args)
{
	cl_uint count{0};
	cl_int err = getIDsFunc(args..., 0, nullptr, &count);

	if(err == CL_DEVICE_NOT_FOUND)
	{
		return {};
	}
	OCL_SAFE_CALL(err);

	std::vector<T> res(count);
	if(count > 0)
	{
		OCL_SAFE_CALL(getIDsFunc(args..., count, res.data(), nullptr));
	}
	return res;
}

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init()) throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:

	auto platforms = getVectorIDs<cl_platform_id>(clGetPlatformIDs);
	std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

	for(int platformIndex = 0; platformIndex < platforms.size(); ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platforms.size() << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		// size_t platformNameSize = 0;
		// OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
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

		//OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:

		// clGetPlatformInfo(...);

		std::cout << "    Platform name: " << getInfoString(clGetPlatformInfo, platform, CL_PLATFORM_NAME) << std::endl;

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы

		std::cout << "    Vendor name: " << getInfoString(clGetPlatformInfo, platform, CL_PLATFORM_VENDOR) << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")

		auto devices = getVectorIDs<cl_device_id>(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL);

		for(int deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			cl_device_id device = devices[deviceIndex];
			std::cout << "\tDevice #" << (deviceIndex + 1) << "/" << devices.size() << std::endl;

			std::cout << "\t    Device name: " << getInfoString(clGetDeviceInfo, device, CL_DEVICE_NAME) << std::endl;

			auto deviceType = getDeviceValue<CL_DEVICE_TYPE>(clGetDeviceInfo, device);

			std::cout << "\t    Device type:";
			if(deviceType & CL_DEVICE_TYPE_CPU) std::cout << " CPU";
			if(deviceType & CL_DEVICE_TYPE_GPU) std::cout << " GPU";
			if(deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << " Accelerator";
			if(deviceType & CL_DEVICE_TYPE_DEFAULT) std::cout << " Default";
			std::cout << std::endl;

			auto globalMemSize = getDeviceValue<CL_DEVICE_GLOBAL_MEM_SIZE>(clGetDeviceInfo, device);
			std::cout << "\t    Global memory size: " << (globalMemSize >> 20) << "MB" << std::endl;

			auto maxClockFreq = getDeviceValue<CL_DEVICE_MAX_CLOCK_FREQUENCY>(clGetDeviceInfo, device);
			std::cout << "\t    Max clock frequency: " << maxClockFreq << "MHz" << std::endl;

			auto maxComputeUnits = getDeviceValue<CL_DEVICE_MAX_COMPUTE_UNITS>(clGetDeviceInfo, device);
			std::cout << "\t    Max compute units: " << maxComputeUnits << std::endl;

		}
	}

	return 0;
}