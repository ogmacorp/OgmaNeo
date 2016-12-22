<!---
  OgmaNeo
  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# OgmaNeo

## Introduction

Welcome to [Ogma](https://ogmacorp.com) OgmaNeo library. A C++ library that contains implementation(s) of Online Predictive Hierarchies, as described in the arXiv.org paper: [Feynman Machine: The Universal Dynamical Systems Computer](http://arxiv.org/abs/1609.03971).

The current release of this library contains a form of Sparse Predictive Hierarchies. Refer to the arXiv.org paper for further details.

Three other OgmaCorp GitHub repositories contain:
- Examples using this library ([OgmaNeoDemos](https://github.com/ogmacorp/OgmaNeoDemos))
- Python bindings to this library ([PyOgmaNeo](https://github.com/ogmacorp/PyOgmaNeo))
- Java (JNI) bindings to this library ([JOgmaNeo](https://github.com/ogmacorp/JOgmaNeo))

## Overview

Refer to the [OgmaNeoDemos](https://github.com/ogmacorp/OgmaNeoDemos) repository for more complicated usage.

OgmaNeo is a fully online learning algorithm, so data must be passed in an appropriately streamed fashion.

The simplest usage of the predictive hierarchy involves calling:

```cpp
    #include <neo/Architect.h>
    #include <neo/Hierarchy.h>

	using namespace ogmaneo;

    // Create the Resources and main ComputeSystem
    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    res->create(ogmaneo::ComputeSystem::_gpu);

    // Use the Architect to build the desired hierarchy
    ogmaneo::Architect arch;
    arch.initialize(1234, res);

    // 1 input layer
    arch.addInputLayer(ogmaneo::Vec2i(4, 4))
        .setValue("in_p_alpha", 0.02f)
        .setValue("in_p_radius", 16);

    // 3 layers using chunk encoders
    for (int l = 0; l < 3; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(36, 36), ogmaneo::_chunk)
        .setValue("sfc_chunkSize", ogmaneo::Vec2i(6, 6))
        .setValue("sfc_ff_radius", 12)
        .setValue("hl_poolSteps", 2)
        .setValue("p_alpha", 0.08f)
        .setValue("p_beta", 0.16f)
        .setValue("p_radius", 12);

    // Generate the hierarchy
    std::shared_ptr<ogmaneo::Hierarchy> hierarchy = arch.generateHierarchy();

    // Input and prediction fields (4x4 size)
    ogmaneo::ValueField2D inputField(ogmaneo::Vec2i(4, 4));
    ogmaneo::ValueField2D predField(ogmaneo::Vec2i(4, 4));
```

You can then step the simulation with:

```cpp
    // Fill the inputField and step the simulation
    hierarchy->simStep(std::vector<ogmaneo::ValueField2D>{ inputField }, true);

	// Retrieve the prediction (same dimensions as the input field)
    predField = hierarchy->getPredictions().front();
```

## Parameters

The OgmaNeo Architect interface has several adjustable parameters.

Parameters are adjusted by performing value changes on the parameter modifier returned by layer add calls.

```cpp
    // 1 input layer
    arch.addInputLayer(ogmaneo::Vec2i(4, 4))
        .setValue("in_p_alpha", 0.02f)
        .setValue("in_p_radius", 16);

    // 3 layers using chunk encoders
    for (int l = 0; l < 3; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(36, 36), ogmaneo::_chunk)
        .setValue("sfc_chunkSize", ogmaneo::Vec2i(6, 6))
        .setValue("sfc_ff_radius", 12)
        .setValue("hl_poolSteps", 2)
        .setValue("p_alpha", 0.08f)
        .setValue("p_beta", 0.16f)
        .setValue("p_radius", 12);
```

Parameters are grouped by (possibly several) prefixes. Below is a list of parameters:

Global (additional parameters, prefix 'ad'):
 - ad_initWeightRange (float, float): global weight initialization range, used when no other ranges are available.
 
Hierarchy layers (prefix 'hl'):
 - hl_poolSteps (int): Number of steps to perform temporal pooling over, 1 means no pooling.

Predictor (prefix 'p'):
 - p_alpha (float): Learning rate.
 - p_radius (int): Input field radius (onto hidden layers).

Input layers (prefix 'in'):
 - Prediction (readout) layers (prefix 'p'):
    - in_p_alpha (float): Learning rate.
    - in_p_radius (int): Input field radius (onto first hidden layer).

Agent layers (prefix 'a'):
 - a_radius (int): Input field radius (onto previous action layer).
 - a_qAlpha (float): Q learning rate.
 - a_qGamma (float): Q gamma (discount factor).
 - a_qLambda (float): Q lambda (trace decay factor).
 - a_epsilon (float): Random threshold for action exploration.
 - a_chunkSize (int, int): Size of a chunk.
 - a_chunkGamma (float): Falloff (higher means faster) for SOM neighborhood radius.

### Encoders

Sparse features STDP (prefix 'sfs'):
 - sfs_inhibitionRadius (int): Inhibitory (lateral) radius.
 - sfs_initWeightRange (float, float): Weight initialization range.
 - sfs_biasAlpha (float): Bias learning rate.
 - sfs_activeRatio (float): Ratio of active units (inverse of sparsity).
 - sfs_gamma (float): State trace decay (used for temporal sparsity management).
 - Feed forward inputs (prefix 'ff'):
    - sfs_ff_radius (int): Radius onto feed forward inputs.
    - sfs_ff_weightAlpha (float): Learning rate for feed forward inputs.
    - sfs_ff_lambda (float): Input trace decay for feed forward inputs.  
 - Recurrent inputs (prefix 'r'):
    - sfs_r_radius (int): Radius onto feed recurrent inputs.
    - sfs_r_weightAlpha (float): Learning rate for recurrent inputs.
    - sfs_r_lambda (float): Input trace decay for recurrent inputs.

Sparse features Delay (prefix 'sfd'):
 - sfd_inhibitionRadius (int): Inhibitory (lateral) radius.
 - sfd_initWeightRange (float, float): Weight initialization range.
 - sfd_biasAlpha (float): Bias learning rate.
 - sfd_activeRatio (float): Ratio of active units (inverse of sparsity).
 - Feed forward inputs (prefix 'ff'):
    - sfd_ff_radius (int): Radius onto feed forward inputs.
    - sfd_ff_weightAlpha (float): Learning rate for feed forward inputs.
    - sfd_ff_lambda (float): Fast synaptic decay for feed forward inputs.
    - sfd_ff_gamma (float): Long synaptic decay for feed forward inputs.

Sparse features Chunk (prefix 'sfc'):
 - sfc_chunkSize (int, int): Size of a chunk.
 - sfc_initWeightRange (float, float): Weight initialization range.
 - sfc_numSamples (int): Number of temporally extended samples (1 means no additional samples).
 - Feed forward inputs (prefix 'ff'):
    - sfc_ff_radius (int): Radius onto feed forward inputs.
    - sfc_ff_weightAlpha (float): Learning rate for feed forward inputs.
    - sfc_ff_lambda (float): Input trace decay for feed forward inputs.  
 - Recurrent inputs (prefix 'r'):
    - sfc_r_radius (int): Radius onto recurrent inputs.
    - sfc_r_weightAlpha (float): Learning rate for recurrent inputs.
    - sfc_r_lambda (float): Input trace decay for recurrent inputs.

Sparse features ReLU (prefix 'sfr')
 - sfr_initWeightRange (float, float): Weight initialization range.
 - sfr_numSamples (int): Number of temporally extended samples (1 means no additional samples).
 - sfr_lateralRadius (int): Inhibitory (lateral) radius.
 - sfr_gamma (float): State trace decay (used for temporal sparsity management).
 - sfr_activeRatio (float): Ratio of active units (inverse of sparsity).
 - sfr_biasAlpha (float): Learning rate of hidden unit biases (for maintaining sparsity).
 - Feed forward (prefix 'ff'):  
    - sfr_ff_radius_hidden (int): Radius onto feed forward oututs.
    - sfr_ff_radius_visible (int): Radius onto feed forward inputs.
    - sfr_ff_weightAlpha_hidden (float): Learning rate for feed forward outputs.
    - sfr_ff_weightAlpha_visible (float): Learning rate for feed forward inputs.
    - sfr_ff_lambda (float): Input trace decay for feed forward inputs.  
 - Recurrent (prefix 'r'):
    - sfr_r_radius_hidden (int): Radius onto recurrent outputs.
    - sfr_r_radius_visible (int): Radius onto recurrent inputs.
    - sfr_r_weightAlpha_hidden (float): Learning rate for recurrent outputs.
    - sfr_r_weightAlpha_visible (float): Learning rate for recurrent inputs.
    - sfr_r_lambda (float): Input trace decay for recurrent inputs.

## Requirements

OgmaNeo requires: a C++1x compiler, [CMake](https://cmake.org/), the [FlatBuffers](https://google.github.io/flatbuffers/) package (version 1.4.0), an OpenCL 1.2 SDK, and the Khronos Group cl2.hpp file.

The library has been tested extensively on:
 - Windows using Microsoft Visual Studio 2013 and 2015,
 - Linux using GCC 4.8 and upwards,
 - Mac OSX using Clang, and
 - Raspberry Pi3, using Raspbian Jessie with GCC 4.8

### CMake

Version 3.1, and upwards, of [CMake](https://cmake.org/) is the required version to use when building the library.

The [CMakeLists.txt](https://github.com/ogmacorp/OgmaNeo/blob/master/CMakeLists.txt) file uses an ExternalProject to download and build the FlatBuffers package. It also defines custom build targets to automatically package kernel code into the library. Kernel packaging is required for [OgmaNeoDemos](https://github.com/ogmacorp/OgmaNeoDemos), and the Python bindings [PyOgmaNeo](https://github.com/ogmacorp/PyOgmaNeo).

### OpenCL

[OpenCL](https://www.khronos.org/opencl/) (Open Compute Language, version 1.2 and upwards) is used to compile, upload and run kernel code on CPU and GPU devices. An OpenCL SDK, with system drivers that support OpenCL 1.2, is required to build and use the OgmaNeo library.

The open source POCL package ([Portable Computing Language](http://portablecl.org/)) can be used for devices that don't have OpenCL vendor driver support. For example the OgmaNeo library using POCL ([release branch 0.13](https://github.com/pocl/pocl/tree/release_0_13)) has been tested on the Raspberry Pi3 device and Travis-CI service.

### CL2 header file

The Khronos Group [cl2.hpp](http://github.khronos.org/OpenCL-CLHPP/) header file is required when building OgmaNeo. And needs to be placed alongside your OpenCL header files. It can be downloaded from Github https://github.com/KhronosGroup/OpenCL-CLHPP/releases

If the `cl2.hpp` file cannot be found the `CMakeLists.txt` script will download the file and include it within library build process. On all other supported platforms the file requires a manual download and copy to the appropriate location (e.g. directory `/usr/include/CL`).

### Flatbuffers

The [FlatBuffers](https://google.github.io/flatbuffers/) package (version 1.4.0), an efficient cross platform serialization library, is used to load and save OgmaNeo internal data (such as hierarchies and agents).

The OgmaNeo CMake build system uses the `CMake\FindFlatBuffers.cmake` script to find the schema compiler and C++ header file include directory. As well as adding a custom build step to compile schema files (extension `.fbs`) and generate helper header files.

If you do not already have the Flatbuffers package installed, the OgmaNeo CMakeLists.txt script will automatically download and build the package into a `3rdparty` directory local to the build.

## Building

[![Build Status](https://travis-ci.org/ogmacorp/OgmaNeo.svg?branch=master)](https://travis-ci.org/ogmacorp/OgmaNeo)

The following commands can be used to build the OgmaNeo library:

> mkdir build; cd build  
> cmake -DBUILD_SHARED_LIBS=ON ..  
> make  

The `cmake` command can be passed a `CMAKE_INSTALL_PREFIX` to determine where to install the library and header files.  
The `BUILD_SHARED_LIBS` boolean cmake option can be used to create dynamic/shared object library (default is to create a _static_ library).

`make install` can be run to install the library. `make uninstall` can be used to uninstall the library.

On Windows it is recommended to use `cmake-gui` to define which generator to use and specify optional build parameters.

## Contributions

Refer to the [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeo/blob/master/CONTRIBUTING.md) file for information on making contributions to OgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [OGMANEO_LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/OGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

The OgmaNeo library uses the Google [FlatBuffers](http://google.github.io/flatbuffers/) package that is licensed with an Apache License (Version 2.0). Refer to this [LICENSE.txt](https://github.com/google/flatbuffers/blob/master/LICENSE.txt) file for the full licensing text associated with the FlatBuffers package.

OgmaNeo Copyright (c) 2016 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
