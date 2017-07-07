<!---
  OgmaNeo
  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# C# bindings for OgmaNeo

## Introduction

This [SWIG](http://www.swig.org/) binding provides an interface into the C++ library, allowing C# code to gain access to the OgmaNeo CPU and GPU accelerated algorithms.

## Requirements

The same requirements that OgmaNeo has, are required for this binding: a C++1x compiler, [CMake](https://cmake.org/), an OpenCL SDK, and Khronos Group's cl2.hpp file.

Additionally the binding requires installation a C# development environment (for example, Mono or Visual Studio) and [SWIG](http://www.swig.org/) v3+

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via, for example, ```sudo apt-get install swig3.0``` command (or via ```yum```).
- Windows requires installation of SWIG (3+). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).

## Installation

The main OgmaNeo C++ library **must** be built and installed (either local or system wide) before attempting to build this binding. The following example can be used to build the CsOgmaNeo library and supporting C# interface code:

> git clone https://github.com/ogmacorp/ogmaneo.git  
> cd ogmaneo  
> mkdir build; cd build  
> cmake -DCMAKE_INSTALL_PREFIX=../install ..  
> make install  
> cd ../Cs  
> mkdir build; cd build  
> cmake ..  
> make  

This will create a shared library (on Windows `CsOgmaNeo.dll`, Linux and Mac OS X `libCsOgmaNeo.so`) that provides the interface to the OgmaNeo library. As well as supporting C# interface code that can be included in a C# project. *Note* The C# interface code relatively imports the CsOgmaNeo library.

## Importing and Setup

The following example C# code can be found in the `src/com/ogmacorp/OgmaNeo/Example/Program.cs` file.

The CsOgmaNeo module can be imported into C# code using:

```csharp
using ogmaneo;
```

The main interface used to setup OgmaNeo is `Resources`. It is used to setup the OgmaNeo C++ library and OpenCL. For example:
```csharp
Resources res = new Resources();
res.create(ComputeSystem.DeviceType._gpu);
```

The `Architect` interface is used to construct and generate a hierarchy. For example:
```csharp
// Instantiate the Architect and link it with the main Resources interface
Architect arch = new Architect();
arch.initialize(1234, _res);

# Use arch.addInputLayer and arch.addHigherLayer to add layers
...

# Finally the hierachy can be generated
Hierarchy hierarchy = arch.generateHierarchy();
```

Layer properties can be accessed via the `ParameterModifier` returned from `addInputLayer` and `addHigherLayer`. Layer parameters are described in the OgmaNeo [README.md](https://github.com/ogmacorp/OgmaNeo/blob/master/README.md) file. For example:
```csharp
int w = 4;
int h = 4;
ParameterModifier inputParams = arch.addInputLayer(new Vec2i(w, h));
inputParams.setValue("in_p_alpha", 0.02f);
inputParams.setValue("in_p_radius", 16);
```

Input values into the hierarchy are created using a `ValueField2D`. For example:
```csharp
ValueField2D inputField = new ValueField2D(new Vec2i(w, h));

for(int y = 0; y < h; y++) {
    for(int x = 0; x < w; x++) {
        inputField.setValue(new Vec2i(x, y), (y * w) + x);
    }
}

vectorvf inputVector = new vectorvf();
inputVector.add(inputField);
```

`Hierarchy.activate` is used to pass input into the hierarchy and run through one prediction step. Optional learning can then be made with `Hierarchy.learn`. For example:
```csharp
hierarchy.activate(inputVector);
hierarchy.learn(inputVector);
```

`Hierarchy.getPrediction()` is used to obtain predictions after an `activate`, or `learn`, has taken place.
```csharp
ValueField2D prediction = hierarchy.getPredictions().get(0);
```

A hierarchy can be saved and loaded as follows:
```csharp
hierarchy.save(res.getComputeSystem(), "filename.opr");
hierarchy.load(res.getComputeSystem(), "filename.opr");
```

## Contributions

Refer to the OgmaNeo [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeo/blob/master/CONTRIBUTING.md) file for details about contributing to OgmaNeo, and the [Ogma Contributor Agreement](https://ogma.ai/wp-content/uploads/2016/09/OgmaContributorAgreement.pdf).

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [OGMANEO_LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/CSOGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmanNeo Copyright (c) 2016-2017 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
