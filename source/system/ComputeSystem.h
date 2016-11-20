// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <system/Uncopyable.h>

//#define CL_HPP_MINIMUM_OPENCL_VERSION 200
//#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#define SYS_DEBUG

#define SYS_ALLOW_CL_GL_CONTEXT 0

namespace ogmaneo {
    /*!
    \brief Compute system
    Holds OpenCL platform, device, context, and command queue
    */
    class ComputeSystem : private Uncopyable {
    public:
        /*!
        \brief OpenCL device types
        */
        enum DeviceType {
            _cpu, _gpu, _all
        };

    private:
        //!@{
        /*!
        \brief OpenCL handles
        */
        cl::Platform _platform;
        cl::Device _device;
        cl::Context _context;
        cl::CommandQueue _queue;
        //!@}

    public:
        /*!
        \brief Create an OpenCL compute system with a given device type.
        Optional: Create from an OpenGL context
        */
        bool create(DeviceType type, bool createFromGLContext = false);

        /*!
        \brief Get underlying OpenCL platform
        */
        cl::Platform &getPlatform() {
            return _platform;
        }

        /*!
        \brief Get underlying OpenCL device
        */
        cl::Device &getDevice() {
            return _device;
        }

        /*!
        \brief Get underlying OpenCL context
        */
        cl::Context &getContext() {
            return _context;
        }

        /*!
        \brief Get underlying OpenCL command queue
        */
        cl::CommandQueue &getQueue() {
            return _queue;
        }
    };
}