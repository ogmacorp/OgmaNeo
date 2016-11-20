// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <system/ComputeSystem.h>

#include <assert.h>

namespace ogmaneo {
    /*!
    \brief Possible encoder identifiers
    */
    enum SparseFeaturesType {
        _stdp, _delay, _chunk
    };

    /*!
    \brief Compute program.
    Holds OpenCL compute program with their associated kernels.
    */
    class ComputeProgram {
    private:
        /*!
        \brief OpenCL program
        */
        cl::Program _program;

        /*!
        \brief Load kernel code from a string
        */
        bool loadFromString(const std::string& kernel, ComputeSystem &cs);

    public:
        /*!
        \brief Load kernel code from a file
        */
        bool loadFromFile(const std::string &name, ComputeSystem &cs);

        /*!
        \brief Load hierarchy default packaged kernel
        */
        bool loadHierarchyKernel(ComputeSystem &cs);

        /*!
        \brief Load hierarchy default packaged kernel
        */
        bool loadPredictorKernel(ComputeSystem &cs);

        /*!
        \brief Load hierarchy default packaged kernel
        */
        bool loadAgentSwarmKernel(ComputeSystem &cs);

        /*!
        \brief Loader for different sparse features (encoders)
        */
        bool loadSparseFeaturesKernel(ComputeSystem &cs, SparseFeaturesType type);

        /*!
        \brief Load extras default packaged kernel
        */
        bool loadExtraKernel(ComputeSystem &cs);

        /*!
        \brief Get the underlying OpenCL program
        */
        cl::Program &getProgram() {
            return _program;
        }
    };
}