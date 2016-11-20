// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ComputeSystem.h"

#include <iostream>

using namespace ogmaneo;

bool ComputeSystem::create(DeviceType type, bool createFromGLContext) {
    std::vector<cl::Platform> allPlatforms;
    cl::Platform::get(&allPlatforms);

    if (allPlatforms.empty()) {
#ifdef SYS_DEBUG
        std::cout << "No platforms found. Check your OpenCL installation." << std::endl;
#endif
        return false;
    }
#ifdef SYS_DEBUG
    std::cout << allPlatforms.size() <<  " platforms found." << std::endl;
#endif

    _platform = allPlatforms.front();

#ifdef SYS_DEBUG
    std::cout << "Using platform: " << _platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
#endif

    std::vector<cl::Device> allDevices;

    switch (type) {
    case _cpu:
        _platform.getDevices(CL_DEVICE_TYPE_CPU, &allDevices);
        break;
    case _gpu:
        _platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
        break;
    case _all:
        _platform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
        break;
    }

    if (!allDevices.empty()) {
        _device = allDevices.front();

        std::vector<size_t> workItemSizes;
        _device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &workItemSizes);

        // Catch incompatible work item sizes (e.g. Apple _cpu device that uses [1024,1,1])
        if (workItemSizes[0] <= 1 || workItemSizes[1] <= 1 || workItemSizes[2] <= 1) {
#ifdef SYS_DEBUG
            std::cerr << "Incompatible device type (unsupported work item size)." << std::endl;
#endif

            if (type != _gpu) {
#ifdef SYS_DEBUG
                std::cout << "Fallback to trying a _gpu device type." << std::endl;
#endif
                _platform.getDevices(CL_DEVICE_TYPE_GPU, &allDevices);
            }
            else {
#ifdef SYS_DEBUG
                std::cerr << "Requested DeviceType is not compatible." << std::endl;
#endif
                return false;
            }
        }
    }

    if (allDevices.empty()) {
#ifdef SYS_DEBUG
        std::cout << "No devices found. Check your OpenCL installation." << std::endl;
#endif
        return false;
    }

    _device = allDevices.front();

#ifdef SYS_DEBUG
    std::cout << "Using device: " << _device.getInfo<CL_DEVICE_NAME>() << std::endl;
#endif

#if(SYS_ALLOW_CL_GL_CONTEXT)
    if (createFromGLContext) {
#if defined (__APPLE__) || defined(MACOSX)
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
            0
        };
#else
#if defined WIN32
        cl_context_properties props[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)static_cast<cl_platform_id>(_platform()),
            0
        };
#else
        cl_context_properties props[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)static_cast<cl_platform_id>(_platform()),
            0
        };
#endif
#endif

        _context = cl::Context(_device, props);
    }
    else
#endif
        _context = _device;

    _queue = cl::CommandQueue(_context, _device);

    return true;
}