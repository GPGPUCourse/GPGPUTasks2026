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

template<typename F, typename T, typename REQ_ID>
std::vector<unsigned char> getStringValue(F f, T id, REQ_ID req_id)
{
	// Враппер функция для f, которая возвращает ответ в виде строки char*:
	// 1 для получения размера результата.
	// 2 для получения самого значения.
	// Параметры:
	// F f: функция, которую надо исполнить
	// T id: айди устройства, для которого надо получить информацию при вызове f
	// REQ_ID req_id: константа запроса, которая передается в вызов f
	// Возвращает строку -- результат исполнения f
	size_t size;
	OCL_SAFE_CALL(f(id, req_id, 0, nullptr, &size));
	std::vector<unsigned char> buffer(size);
	OCL_SAFE_CALL(f(id, req_id, size, buffer.data(), nullptr));
	return buffer;
}

template<typename REQ_TYPE, typename F, typename T, typename REQ_ID>
REQ_TYPE getNonStringValue(F f, T id, REQ_ID req_id)
{
	// Враппер для функции f, которая возвращает один элемент а не char*
	// Параметры:
	// REQ_TYPE: тип возвращаемого значения
	// F f: функция, которую надо исполнить
	// T id: айди устройства, для которого надо получить информацию при вызове f
	// REQ_ID req_id: константа запроса, которая передается в вызов f
	// Возвращает:
	// REQ_TYPE value: результат вызова f
	REQ_TYPE value;
	OCL_SAFE_CALL(f(id, req_id, sizeof(REQ_TYPE), &value, nullptr));
	return value;
}

std::string getDeviceTypeName(cl_device_type type) {
	std::string name = "";
	if (type & CL_DEVICE_TYPE_CPU)
		name += "CPU ";
	if (type & CL_DEVICE_TYPE_GPU)
		name += "GPU ";
	if (type & CL_DEVICE_TYPE_ACCELERATOR)
		name += "Accelerator ";
	if (type & CL_DEVICE_TYPE_CUSTOM)
		name += "Custom ";
	if (type & CL_DEVICE_TYPE_DEFAULT)
		name += "Default ";

	if (name.empty()) name = "Unknown ";

	name.pop_back();
	return name;
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
		auto vendor_name = getStringValue(clGetPlatformInfo, platform, CL_PLATFORM_VENDOR);
		std::cout << "    Vendor name: " << vendor_name.data() << std::endl;

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
		std::cout << "    Number of devices: " << devicesCount << std::endl;

		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			std::cout << "        Device #" << (deviceIndex + 1) << "/" << devicesCount << ":" << std::endl;
			cl_device_id device = devices[deviceIndex];
			auto device_name = getStringValue(clGetDeviceInfo, device, CL_DEVICE_NAME);
			std::cout << "            Device name: " << device_name.data() << std::endl;

			std::cout << "            Device type: " <<
				getDeviceTypeName(getNonStringValue<cl_device_type>(clGetDeviceInfo, device, CL_DEVICE_TYPE)) << std::endl;

			std::cout << "            Device size memory: " <<
				(getNonStringValue<cl_ulong>(clGetDeviceInfo, device, CL_DEVICE_GLOBAL_MEM_SIZE) >> 20) << " MB" << std::endl;

			std::cout << "            Device cache line size: " <<
				getNonStringValue<cl_uint>(clGetDeviceInfo, device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << " B" << std::endl;

			std::cout << "            Device max freq in MHz: " <<
				getNonStringValue<cl_uint>(clGetDeviceInfo, device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
		}
	}

	return 0;
}
