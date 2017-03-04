<!---
  OgmaNeo
  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# OgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby) [![Build Status](https://travis-ci.org/ogmacorp/OgmaNeo.svg?branch=master)](https://travis-ci.org/ogmacorp/OgmaNeo)

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
    hierarchy->activate(std::vector<ogmaneo::ValueField2D>{ inputField });

    hierarchy->learn(std::vector<ogmaneo::ValueField2D>{ inputField });

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
 - p_alpha (float): Current layer prediction learning rate.
 - p_beta (float): Feed back learning rate.
 - p_radius (int): Input field radius (onto hidden layers).
 - p_lambda (int): TD lambda, for reinforcement learning (if enabled).

### Encoders

Sparse features Chunk (prefix 'sfc'):
 - sfc_chunkSize (int, int): Size of a chunk.
 - sfc_gamma (float): Small boosting factor.
 - sfc_initWeightRange (float, float): Weight initialization range.
 - Feed forward inputs (prefix 'ff'):
    - sfc_ff_numSamples (int): Number of temporally extended samples (1 means no additional samples). Should be around 2 * hl_poolsteps
    - sfc_ff_radius (int): Radius onto feed forward inputs.
    - sfc_ff_weightAlpha (float): Learning rate for feed forward inputs.
    - sfc_ff_lambda (float): Input trace decay for feed forward inputs.  

Sparse features Distance (prefix 'sfd'):
 - sfd_chunkSize (int, int): Size of a chunk.
 - sfd_gamma (float): Decay of inverse learning rates. Should be below but close to 1.
 - sfd_initWeightRange (float, float): Weight initialization range.
 - Feed forward inputs (prefix 'ff'):
    - sfd_ff_numSamples (int): Number of temporally extended samples (1 means no additional samples). Should be around 2 * hl_poolsteps
    - sfd_ff_radius (int): Radius onto feed forward inputs.
    - sfd_ff_weightAlpha (float): Learning rate for feed forward inputs.
    - sfd_ff_lambda (float): Input trace decay for feed forward inputs.  

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
