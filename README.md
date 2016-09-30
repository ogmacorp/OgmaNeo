<!---
  OgmaNeo
  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# OgmaNeo

## Introduction

Welcome to [Ogma Intelligent Systems Corp](https://ogmacorp.com) OgmaNeo library. A C++ library that contains implementation(s) of Online Predictive Hierarchies, as described in the arXiv.org paper: [Feynman Machine: The Universal Dynamical Systems Computer](http://arxiv.org/abs/1609.03971).

The current release of this library contains a form of Sparse Predictive Hierarchies. Refer to the arXiv.org paper for further details.

Two other OgmaCorp GitHub repositories contain:
- Examples using this library ([OgmaNeoDemos](https://github.com/ogmacorp/OgmaNeoDemos)), and
- Python bindings to this library ([PyOgmaNeo](https://github.com/ogmacorp/PyOgmaNeo)).

## Overview

Refer to the [OgmaNeoDemos](https://github.com/ogmacorp/OgmaNeoDemos) repository for more complicated usage.

OgmaNeo is a fully online learning algorithm, so data must be passed in an appropriately streamed fashion.

The simplest usage of the predictive hierarchy involves calling:

```cpp
	#include <OgmaNeo.h>
	using namespace ogmaneo;

	// Define a pseudo random number generator
	std::mt19937 generator(time(nullptr));

	// Create the main compute system interface
	ComputeSystem cs;
	cs.create(ComputeSystem::_gpu);

	// Load the main kernel code
	ComputeProgram prog;
	prog.loadMainKernel(cs);

	// --------------------------- Create the Predictive Hierarchy ---------------------------

	// Temporary input buffers
	cl::Image2D inputImage = cl::Image2D(
		cs.getContext(), CL_MEM_READ_WRITE,
		cl::ImageFormat(CL_R, CL_FLOAT), 2, 2);

	cl::Image2D inputImageCorrupted = cl::Image2D(
		cs.getContext(), CL_MEM_READ_WRITE,
		cl::ImageFormat(CL_R, CL_FLOAT), 2, 2);

	// Layer descriptors for hierarchy
	std::vector<FeatureHierarchy::LayerDesc> layerDescs(3);
	std::vector<Predictor::PredLayerDesc> pLayerDescs(3);

	// Alter layer descriptors
	layerDescs[0]._size = { 64, 64 };
	layerDescs[1]._size = { 64, 64 };
	layerDescs[2]._size = { 64, 64 };

	Predictor ph;

	// 2x2 input field
	ph.createRandom(cs, prog, { 2, 2 }, pLayerDescs, layerDescs, { -0.01f, 0.01f }, 0.0f, generator);
```

You can then step the simulation with:

```cpp
	cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
	cl::array<cl::size_type, 3> region = { 2, 2, 1 };
	
	// Copy values into the temporary OpenCL image buffer
	cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, origin,region, 0,0, values.data());

	// Copy corrupted values into the temporary OpenCL image buffer
	cs.getQueue().enqueueWriteImage(inputImageCorrupted, CL_TRUE, origin,region, 0,0, valuesCorrupted.data());

	// Step the simulation
	ph.simStep(cs, inputImage, inputImageCorrupted, generator, true);

	// Retrieve the prediction (same dimensions as the input field)
	std::vector<float> pred(4);
	cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, origin,region, 0,0, pred.data());
```

## Parameters

The two main classes (Predictor and AgentSwarm) have several adjustable parameters. Below is a brief description of each as it appears in the simplified interface (LayerDescs.h).

- LayerDescs

	- width, height: (int) size of layer
	
	Feature hierarchy parameters

	- feedForwardRadius: (int) radius of nodes onto the inputs
	- recurrentRadius: (int) radius onto previous timestep layer states
	- inhibitionRadius: (int) radius across a layer (inhibition)
	- spFeedForwardWeightAlpha: (float) learning rate of feed forward weights
	- spRecurrentWeightAlpha: (float) learning rate of recurrent weights
	- spBiasAlpha: (float) learning rate of biases
	- spActiveRatio: (float) ratio of active units within a layer (< 0.5 = sparse, 0.5 = dense)

	Predictor layer parameters

	- predRadius: (int) radius onto current and higher layers for prediction
	- predAlpha: (float) learning rate of current layer predictive weights
	- predBeta: (float) learning rate of higher layer predictive weights
	
	Agent layer parameters

	- qRadius: (int) radius onto previous agent layers (modulated by the feature hierarchy)
	- qAlpha: (float) learning rate of Q values
	- qGamma: (float) discount factor of Q values
	- qLambda: (float) trace decay of Q values (1 = no decay, 0 = instant decay)
	- epsilon: (float) exploration ratio (0 = no exploration, 1 = only exploration)

## Requirements

OgmaNeo requires: a C++1x compiler, [CMake](https://cmake.org/), the [FlatBuffers](https://google.github.io/flatbuffers/) package (version 1.4.0), an OpenCL 1.2 SDK, and the Khronos Group cl2.hpp file.

The library has been tested extensively on Windows using Microsoft Visual Studio 2013 and 2015, and on Linux using GCC 4.8 and upwards.

### CMake

Version 3.1, and upwards, of [CMake](https://cmake.org/) is the required version to use when building the library.

The [CMakeLists.txt](https://github.com/ogmacorp/OgmaNeo/blob/master/CMakeLists.txt) file uses an ExternalProject to download and build the FlatBuffers package. It also defines custom build targets to automatically package kernel code into the library. Kernel packaging is required for [OgmaNeoDemos](https://github.com/ogmacorp/OgmaNeoDemos), and the Python bindings [PyOgmaNeo](https://github.com/ogmacorp/PyOgmaNeo).

### OpenCL

[OpenCL](https://www.khronos.org/opencl/) (Open Compute Language, version 1.2 and upwards) is used to compile, upload and run kernel code on CPU and GPU devices. An OpenCL SDK, with system drivers that support OpenCL 1.2, is required to build and use the OgmaNeo library.

The open source POCL package ([Portable Computing Language](http://portablecl.org/)) can be used for devices that don't have OpenCL vendor driver support. For example the OgmaNeo library using POCL ([release branch 0.13](https://github.com/pocl/pocl/tree/release_0_13)) has been tested on the Raspberry Pi3.

### CL2 header file

The Khronos Group [cl2.hpp](http://github.khronos.org/OpenCL-CLHPP/) header file is required when building OgmaNeo. And needs to be placed alongside your OpenCL header files. It can be downloaded from Github https://github.com/KhronosGroup/OpenCL-CLHPP/releases

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

On Windows it is recommended to use `cmake-gui` to define which generator to use and specify option build parameters.

## Contributions

Refer to the [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeo/blob/master/CONTRIBUTING.md) file for information on making contributions to OgmaNeo.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [OGMANEO_LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/OGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma Intelligent Systems Corp [licenses@ogmacorp.com](licenses@ogmacorp.com) to discuss commercial use and licensing options.

The OgmaNeo library uses the Google [FlatBuffers](http://google.github.io/flatbuffers/) package that is licensed with an Apache License (Version 2.0). Refer to this [LICENSE.txt](https://github.com/google/flatbuffers/blob/master/LICENSE.txt) file for the full licensing text associated with the FlatBuffers package.

OgmaNeo Copyright (c) 2016 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
