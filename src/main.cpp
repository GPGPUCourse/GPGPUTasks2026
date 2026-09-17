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
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// DONE: это #define CL_INVALID_VALUE

		//  1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
		std::vector<unsigned char> platformName(platformNameSize, 0);
	    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		//  1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы
	    size_t vendornamesize=0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform,CL_PLATFORM_VENDOR,0,nullptr,&vendornamesize));
	    std::vector<unsigned char> vendorName(vendornamesize,0);
	    OCL_SAFE_CALL(clGetPlatformInfo(platform,CL_PLATFORM_VENDOR,vendornamesize,vendorName.data(),nullptr));
	    std::cout<<"    Vendor name: "<<vendorName.data()<< std::endl;
		// 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
		cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,0,nullptr,&devicesCount));
	    std::cout << "    Number of OpenCL devices: " << devicesCount << std::endl;
	    std::vector<cl_device_id> devices(devicesCount);
	    OCL_SAFE_CALL(clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,devicesCount,devices.data(),nullptr));
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// 2.2
			cl_device_id device = devices[deviceIndex];
		    size_t devicenamesize = 0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_NAME,0,nullptr,&devicenamesize));
		    std::vector<unsigned char> devicename(devicenamesize,0);
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_NAME,devicenamesize,devicename.data(),nullptr));
		    std::cout<<"        Device name: "<<devicename.data()<<std::endl;
		    size_t devicetypesize=0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_TYPE,0,nullptr,&devicetypesize));
		    size_t devicetype=0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_TYPE,devicetypesize,&devicetype,nullptr));
		    std::cout<<"        Device type: "<<(devicetype==CL_DEVICE_TYPE_CPU ? "CPU" : devicetype==CL_DEVICE_TYPE_GPU ? " GPU" : devicetype==CL_DEVICE_TYPE_ACCELERATOR ? "ACCELERATOR" : "OTHER")<<std::endl;
			cl_ulong devicememory=0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,16,&devicememory,nullptr));
		    std::cout<<"        Device memory (GB): "<<((double) devicememory)/(1024*1024*1024)<<std::endl;
		    cl_uint cachelinesize=0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,8,&cachelinesize,nullptr));
		    std::cout<<"        Device cacheline size (bytes): "<<cachelinesize<<std::endl;
		    cl_ulong localmemory=0;
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_LOCAL_MEM_SIZE,16,&localmemory,nullptr));
		    std::cout<<"        Device local memory (KB): "<<((double) localmemory)/(1024)<<std::endl;
		    cl_bool iscompileravailable=false; ///8 байт это uint !
		    OCL_SAFE_CALL(clGetDeviceInfo(device,CL_DEVICE_COMPILER_AVAILABLE,8,&iscompileravailable,nullptr));
		    std::cout<<"        Device compiler available: "<<(iscompileravailable ? "True" : "False")<<std::endl;
		    // Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
		}
	}

	return 0;
}
/*
Я крутой я запускаюсь на сервере! 1111111
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: NVIDIA CUDA
    Vendor name: NVIDIA Corporation
    Number of OpenCL devices: 5
        Device name: NVIDIA A100-SXM4-80GB
        Device type:  GPU
        Device memory (GB): 79.2498
        Device cacheline size (bytes): 128
        Device local memory (KB): 48
        Device compiler available: True
        Device name: NVIDIA A100-SXM4-80GB
        Device type:  GPU
        Device memory (GB): 79.2498
        Device cacheline size (bytes): 128
        Device local memory (KB): 48
        Device compiler available: True
        Device name: NVIDIA A100-SXM4-80GB
        Device type:  GPU
        Device memory (GB): 79.2498
        Device cacheline size (bytes): 128
        Device local memory (KB): 48
        Device compiler available: True
        Device name: NVIDIA A100-SXM4-80GB
        Device type:  GPU
        Device memory (GB): 79.2498
        Device cacheline size (bytes): 128
        Device local memory (KB): 48
        Device compiler available: True
        Device name: NVIDIA DGX Display
        Device type:  GPU
        Device memory (GB): 3.62762
        Device cacheline size (bytes): 128
        Device local memory (KB): 48
        Device compiler available: True
*/