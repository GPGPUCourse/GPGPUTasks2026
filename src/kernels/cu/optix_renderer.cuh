#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

#if __has_include(<optix.h>)
#include "optix_params.cuh"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

static void check_optix(OptixResult result)
{
    if (result != OPTIX_SUCCESS)
        throw ::std::runtime_error(optixGetErrorName(result));
}

struct OptixRenderer {
    OptixDeviceContext context{};
    OptixModule module{};
    OptixProgramGroup groups[3]{};
    OptixPipeline pipeline{};

    ~OptixRenderer() {
        if (pipeline) optixPipelineDestroy(pipeline);
        for (auto group : groups)
            if (group) optixProgramGroupDestroy(group);
        if (module) optixModuleDestroy(module);
        if (context) optixDeviceContextDestroy(context);
    }
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) ShaderRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
#endif

namespace cuda {
bool render_optix(const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
                  const CameraViewGPU& camera, gpu::gpu_mem_32i& ids, gpu::gpu_mem_32f& ao, gpu::gpu_mem_32f& depth)
{
    const char* path = ::std::getenv("TASK08_OPTIX_PTX");
    if (!path) return false;
#if __has_include(<optix.h>)
    // NVIDIA walks through triangle GAS construction, pipeline setup, shader binding records and launch. https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/
    OptixRenderer rt;
    check_optix(optixInit());
    check_optix(optixDeviceContextCreate(nullptr, nullptr, &rt.context));
    unsigned version = 0;
    check_optix(optixDeviceContextGetProperty(rt.context, OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                                             &version, sizeof(version)));
    if (!version) return false;
    ::std::ifstream file(path, ::std::ios::binary);
    if (!file) throw ::std::runtime_error("Unable to open TASK08_OPTIX_PTX");
    ::std::string ptx(::std::istreambuf_iterator<char>(file), {});
    OptixPipelineCompileOptions options{};
    options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    options.numPayloadValues = 2;
    options.numAttributeValues = 2;
    options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    options.pipelineLaunchParamsVariableName = "params";
    OptixModuleCompileOptions moduleOptions{};
    check_optix(optixModuleCreate(rt.context, &moduleOptions, &options, ptx.data(), ptx.size(),
                                 nullptr, nullptr, &rt.module));
    OptixProgramGroupDesc programs[3]{};
    programs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    programs[0].raygen = {rt.module, "__raygen__render_optix"};
    programs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programs[2].hitgroup.moduleCH = rt.module;
    programs[2].hitgroup.entryFunctionNameCH = "__closesthit__triangle";
    programs[2].hitgroup.moduleAH = rt.module;
    programs[2].hitgroup.entryFunctionNameAH = "__anyhit__shadow";
    OptixProgramGroupOptions groupOptions{};
    check_optix(optixProgramGroupCreate(rt.context, programs, 3, &groupOptions, nullptr, nullptr, rt.groups));
    OptixPipelineLinkOptions link{};
    link.maxTraceDepth = 1;
    check_optix(optixPipelineCreate(rt.context, &options, &link, rt.groups, 3, nullptr, nullptr, &rt.pipeline));
    OptixStackSizes stack{};
    ShaderRecord records[3]{};
    for (int i = 0; i < 3; ++i) {
        check_optix(optixUtilAccumulateStackSizes(rt.groups[i], &stack, rt.pipeline));
        check_optix(optixSbtRecordPackHeader(rt.groups[i], &records[i]));
    }
    unsigned traversal, state, continuation;
    check_optix(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation));
    check_optix(optixPipelineSetStackSize(rt.pipeline, traversal, state, continuation, 1));
    gpu::shared_device_buffer_typed<ShaderRecord> deviceRecords(3);
    deviceRecords.writeN(records, 3);
    auto base = reinterpret_cast<CUdeviceptr>(deviceRecords.cuptr());
    OptixShaderBindingTable sbt{};
    sbt.raygenRecord = base;
    sbt.missRecordBase = base + sizeof(ShaderRecord);
    sbt.hitgroupRecordBase = base + 2 * sizeof(ShaderRecord);
    sbt.missRecordCount = sbt.hitgroupRecordCount = 1;
    sbt.missRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes = sizeof(ShaderRecord);

    CUdeviceptr vertexBuffer = reinterpret_cast<CUdeviceptr>(vertices.cuptr());
    unsigned flags = OPTIX_GEOMETRY_FLAG_NONE;
    // https://github.com/ingowald/optix7course/blob/6fafedcddac6df2ee367ed7eb08c8a982d1db069/example04_firstTriangleMesh/SampleRenderer.cpp#L131-L223
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto& mesh = input.triangleArray;
    mesh.vertexBuffers = &vertexBuffer;
    mesh.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    mesh.numVertices = vertices.number() / 3;
    mesh.indexBuffer = reinterpret_cast<CUdeviceptr>(faces.cuptr());
    mesh.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    mesh.numIndexTriplets = faces.number() / 3;
    mesh.flags = &flags;
    mesh.numSbtRecords = 1;
    OptixAccelBuildOptions build{};
    build.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    build.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    check_optix(optixAccelComputeMemoryUsage(rt.context, &build, &input, 1, &sizes));
    gpu::gpu_mem_8u scratch(sizes.tempSizeInBytes), gas(sizes.outputSizeInBytes);
    auto stream = gpu::Context().cudaStream();
    OptixTraversableHandle handle;
    check_optix(optixAccelBuild(rt.context, stream, &build, &input, 1,
        reinterpret_cast<CUdeviceptr>(scratch.cuptr()), scratch.number(),
        reinterpret_cast<CUdeviceptr>(gas.cuptr()), gas.number(), &handle, nullptr, 0));
    OptixRenderParams params{handle, vertices.cuptr(), faces.cuptr(), camera, ids.cuptr(), ao.cuptr(), depth.cuptr()};
    gpu::shared_device_buffer_typed<OptixRenderParams> deviceParams(1);
    deviceParams.writeN(&params, 1);
    CUDA_CHECK_KERNEL_SYNC(stream);
    auto start = ::std::chrono::steady_clock::now();
    check_optix(optixLaunch(rt.pipeline, stream, reinterpret_cast<CUdeviceptr>(deviceParams.cuptr()),
                           sizeof(params), &sbt, camera.K.width, camera.K.height, 1));
    CUDA_CHECK_KERNEL_SYNC(stream);
    ::std::chrono::duration<double, ::std::milli> elapsed = ::std::chrono::steady_clock::now() - start;
    ::std::cout << "OptiX render time: " << elapsed.count() << " ms" << ::std::endl;
    return true;
#else
    throw ::std::runtime_error("Build with the OptiX SDK include directory in CMAKE_CUDA_FLAGS");
#endif
}
}
