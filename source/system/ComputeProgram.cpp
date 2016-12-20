// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ComputeProgram.h"

#include <fstream>
#include <iostream>
#include <numeric>

#include "kernels/neoKernelsHierarchy.h"
#include "kernels/neoKernelsPredictor.h"
#include "kernels/neoKernelsAgentSwarm.h"
#include "kernels/neoKernelsExtra.h"

// Add additional encoders here
#include "kernels/neoKernelsSparseFeaturesChunk.h"
#include "kernels/neoKernelsSparseFeaturesDelay.h"
#include "kernels/neoKernelsSparseFeaturesSTDP.h"
#include "kernels/neoKernelsSparseFeaturesReLU.h"

using namespace ogmaneo;

bool ComputeProgram::loadHierarchyKernel(ComputeSystem &cs) {
    std::string kernel = std::accumulate(
        neoKernelsHierarchy_ocl, neoKernelsHierarchy_ocl + sizeof(neoKernelsHierarchy_ocl) / sizeof(neoKernelsHierarchy_ocl[0]),
        std::string(""));

    return loadFromString(kernel, cs);
}

bool ComputeProgram::loadPredictorKernel(ComputeSystem &cs) {
    std::string kernel = std::accumulate(
        neoKernelsPredictor_ocl, neoKernelsPredictor_ocl + sizeof(neoKernelsPredictor_ocl) / sizeof(neoKernelsPredictor_ocl[0]),
        std::string(""));

    return loadFromString(kernel, cs);
}

bool ComputeProgram::loadAgentSwarmKernel(ComputeSystem &cs) {
    std::string kernel = std::accumulate(
        neoKernelsAgentSwarm_ocl, neoKernelsAgentSwarm_ocl + sizeof(neoKernelsAgentSwarm_ocl) / sizeof(neoKernelsAgentSwarm_ocl[0]),
        std::string(""));

    return loadFromString(kernel, cs);
}

bool ComputeProgram::loadExtraKernel(ComputeSystem &cs) {
    std::string kernel = std::accumulate(
        neoKernelsExtra_ocl, neoKernelsExtra_ocl + sizeof(neoKernelsExtra_ocl) / sizeof(neoKernelsExtra_ocl[0]),
        std::string(""));

    return loadFromString(kernel, cs);
}

bool ComputeProgram::loadSparseFeaturesKernel(ComputeSystem &cs, SparseFeaturesType type) {
    std::string kernel;
    
    switch (type) {
    case _stdp:
        kernel = std::accumulate(
            neoKernelsSparseFeaturesSTDP_ocl, neoKernelsSparseFeaturesSTDP_ocl + sizeof(neoKernelsSparseFeaturesSTDP_ocl) / sizeof(neoKernelsSparseFeaturesSTDP_ocl[0]),
            std::string(""));

        break;

    case _delay:
        kernel = std::accumulate(
            neoKernelsSparseFeaturesDelay_ocl, neoKernelsSparseFeaturesDelay_ocl + sizeof(neoKernelsSparseFeaturesDelay_ocl) / sizeof(neoKernelsSparseFeaturesDelay_ocl[0]),
            std::string(""));

        break;

    case _chunk:
        kernel = std::accumulate(
            neoKernelsSparseFeaturesChunk_ocl, neoKernelsSparseFeaturesChunk_ocl + sizeof(neoKernelsSparseFeaturesChunk_ocl) / sizeof(neoKernelsSparseFeaturesChunk_ocl[0]),
            std::string(""));

        break;

    case _ReLU:
        kernel = std::accumulate(
            neoKernelsSparseFeaturesReLU_ocl, neoKernelsSparseFeaturesReLU_ocl + sizeof(neoKernelsSparseFeaturesReLU_ocl) / sizeof(neoKernelsSparseFeaturesReLU_ocl[0]),
            std::string(""));

        break;
    }

    return loadFromString(kernel, cs);
}

bool ComputeProgram::loadFromFile(const std::string &name, ComputeSystem &cs) {
    std::ifstream fromFile(name);

    if (!fromFile.is_open()) {
#ifdef SYS_DEBUG
        std::cerr << "Could not open file " << name << "!" << std::endl;
#endif
        return false;
    }

    std::string kernel = "";

    while (!fromFile.eof() && fromFile.good()) {
        std::string line;

        std::getline(fromFile, line);

        kernel += line + "\n";
    }

    return loadFromString(kernel, cs);
}

bool ComputeProgram::loadFromString(const std::string& kernel, ComputeSystem &cs) {
    _program = cl::Program(cs.getContext(), kernel);

    if (_program.build(std::vector<cl::Device>(1, cs.getDevice())) != CL_SUCCESS) {
#ifdef SYS_DEBUG
        std::cerr << "Error building: " << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cs.getDevice()) << std::endl;
#endif
        return false;
    }

    return true;
}
