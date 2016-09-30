// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <system/SharedLib.h>
#include <system/ComputeSystem.h>
#include <system/ComputeProgram.h>
#include <neo/Helpers.h>

namespace ogmaneo {
    /*!
    \brief Compute system interface (access point for language bindings).
    Contains a ComputeSystem that holds the OpenCL platform, device, context, and command queue.
    */
    class OGMA_API ComputeSystemInterface {
    private:
        /*!
        \brief Compute system.
        Holds OpenCL platform, device, context, and command queue.
        */
        ComputeSystem cs;

    public:
        ComputeSystemInterface() {
        }

        virtual ~ComputeSystemInterface() {
        }

        /*!
        \brief Create an OpenCL compute system with a given device type.
        Optional: Create from an OpenGL context
        */
        bool create(ComputeSystem::DeviceType type, bool createFromGLContext = false) {
            return cs.create(type, createFromGLContext);
        }

        /*!
        \brief Accessor to internal ComputeSystem
        */
        ComputeSystem &operator()() {
            return cs;
        }
    };

    /*!
    \brief Compute program interace (access point for language bindings).
    Contains a ComputeProgram that holds an OpenCL compute program with their associated kernels.
    */
    class OGMA_API ComputeProgramInterface {
    private:
        /*!
        \brief Compute program.
        Holds OpenCL compute program with their associated kernels.
        */
        ComputeProgram prog;

    public:
        ComputeProgramInterface() {
        }

        virtual ~ComputeProgramInterface() {
        }

        /*!
        \brief Load main default packaged kernel
        */
        bool loadMainKernel(ComputeSystemInterface &cs) {
            return prog.loadMainKernel(cs());
        }

        /*!
        \brief Load extras default packaged kernel
        */
        bool loadExtraKernel(ComputeSystemInterface &cs) {
            return prog.loadExtraKernel(cs());
        }

        /*!
        \brief Load kernel code from a file
        */
        bool loadFromFile(const std::string &name, ComputeSystemInterface &cs) {
            return prog.loadFromFile(name, cs());
        }

        /*!
        \brief Accessor to internal ComputeProgram
        */
        ComputeProgram &operator()() {
            return prog;
        }
    };
}