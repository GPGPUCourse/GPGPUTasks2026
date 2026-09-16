#pragma once

#include <CL/cl.h>
#include <string>

namespace cl {
template<cl_device_info>
struct param_traits;

template<>
struct param_traits<CL_DEVICE_AVAILABLE>
{
	using type = cl_bool;
};

template<>
struct param_traits<CL_DEVICE_NAME>
{
	using type = std::string;
};

template<>
struct param_traits<CL_DEVICE_TYPE>
{
	using type = cl_device_type;
};

template<>
struct param_traits<CL_DEVICE_GLOBAL_MEM_SIZE>
{
	using type = cl_ulong;
};

template<>
struct param_traits<CL_DEVICE_HOST_UNIFIED_MEMORY>
{
	using type = cl_bool;
};

template<>
struct param_traits<CL_DEVICE_MAX_CLOCK_FREQUENCY>
{
	using type = cl_uint;
};

template<>
struct param_traits<CL_DEVICE_MAX_COMPUTE_UNITS>
{
	using type = cl_uint;
};

template<cl_device_info info>
using param_traits_t = typename param_traits<info>::type;
}  // namespace cl
