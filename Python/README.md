<!---
  OgmaNeo
  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeo is licensed to you under the terms described
  in the OGMANEO_LICENSE.md file included in this distribution.
--->

# Python bindings for OgmaNeo

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby)

## Introduction

This [SWIG](http://www.swig.org/) binding provides an interface into the C++ library, allowing Python scripts to gain access to the OgmaNeo CPU and GPU accelerated algorithms.

## Requirements

The same requirements that OgmaNeo has, are required for the bindings: a C++1x compiler, [CMake](https://cmake.org/), an OpenCL SDK, and the Khronos Group's cl2.hpp file.

Additionally this binding requires an installation of [SWIG](http://www.swig.org/) v3+

These bindings have been tested using:

| Distribution | Operating System (Compiler) |
| --- | ---:|
| Python 2.7 | Linux (GCC 4.8+) |
| Python 2.7 | Mac OSX |
| Anaconda Python 2.7 3.4 & 3.5 | Linux (GCC 4.8+) |
| Anaconda Python 3.5 | Windows (MSVC 2015) |

Further information on Python compatible Windows compilers can be found [here](https://wiki.python.org/moin/WindowsCompilers).

#### [SWIG](http://www.swig.org/)

- Linux requires SWIG installed via, for example ```sudo apt-get install swig3.0``` command (or via ```yum```).
- Windows requires installation of SWIG (v3). With the SourceForge Zip expanded, and the PATH environment variable updating to include the SWIG installation binary directory (for example `C:\Program Files (x86)\swigwin-3.0.8`).

## Installation

The main OgmaNeo C++ library **must** be built and installed (either local or system wide) before attempting to build this binding. The following example can be used to build the Python package:

> git clone https://github.com/ogmacorp/ogmaneo.git  
> cd ogmaneo  
> mkdir build; cd build  
> cmake -DCMAKE_INSTALL_PREFIX=../install ..  
> make install  
> cd ../Python  
> python3 setup.py install --user  

## Importing and Setup

The OgmaNeo Python module can be imported using:

```python
import ogmaneo
```

The main interface used to setup OgmaNeo is `Resources`. It is used to setup the OgmaNeo C++ library and OpenCL. For example:
```python
res = ogmaneo.Resources()
res.create(ogmaneo.ComputeSystem._gpu)
```

The `Architect` interface is used to construct and generate a hierarchy. For example:
```python
# Instantiate the Architect and link it with the main Resources interface
arch = ogmaneo.Architect()
arch.initialize(seed, res)

# Use arch.addInputLayer and arch.addHigherLayer to add layers
...

# Finally the hierachy can be generated
hierarchy = arch.generateHierarchy()
```

Layer properties can be accessed via the `ParameterModifier` returned from `addInputLayer` and `addHigherLayer`. Layer parameters are described in the OgmaNeo [README.md](https://github.com/ogmacorp/OgmaNeo/blob/master/README.md) file. For example:
```python
w, h = 16, 16
inputParams = arch.addInputLayer(ogmaneo.Vec2i(w, h))
inputParams.setValue("in_p_alpha", 0.02)
inputParams.setValue("in_p_radius", 8)
```

Input values into the hierarchy are created using a `ValueField2D`. For example:
```python
inputField = ogmaneo.ValueField2D(ogmaneo.Vec2i(w, h))

for y in range(h):
    for x in range(w):
        inputField.setValue(ogmaneo.Vec2i(x,y), (y*w)+x)

inputVector = ogmaneo.vectorvf()
inputVector.push_back(inputField)
```

`Hierarchy.activate` is used to pass input into the hierarchy and run through one prediction step. Optional learning can then be made with `Hierarchy.learn`. For example:
```python
hierarchy.activate(inputVector)  
hierarchy.learn(inputVector)  
```

`Hierarchy.getPrediction()` is used to obtain predictions after an `activate`, or `learn`, has taken place.
```python
prediction = hierarchy.getPrediction()[0]
```

A hierarchy can be saved and loaded as follows:
```python
hierarchy.save(res.getComputeSystem(), "filename.opr")  
hierarchy.load(res.getComputeSystem(), "filename.opr")
```

The above example Python code can be found in the `Example.py` file.

## OgmaNeo Developers

By default a CMake library configuration is used to find an existing installation of the OgmaNeo library. If it cannot find the library, the CMakeLists.txt file automatically clones the OgmaNeo master repository and builds the library in place.

Two options exist for OgmaNeo library developers that can redirect this process:

- The `CMakeLists.txt` file can be modified locally to point to a fork of an OgmaNeo repository, and also clone a particular branch from a fork. The `GIT_REPOSITORY` line in `CMakeLists.txt` file can be changed to point to a fork location. An additional `GIT_TAG` line can be added to obtain a particular branch from a fork.

- If you require the use of a local clone of OgmaNeo, the `setup.cfg` file can be modified locally to achieve this. An extra line can be added to specify optional CMake arguments.  
Similar to the following, but with `<repo_dir>` changed to point to your OgmaNeo root directory, or to appropriate system wide locations. This assumes that the OgmaNeo CMAKE_INSTALL_PREFIX has been set to `<repo_dir>/install` and that a `make install` build step has been performed before running the `python setup.py install --user` command. Make sure to use `/` as a path seperator.  
> [build_ext]  
> inplace=0  
> extra-cmake-args=-DOGMANEO_LIBRARY=\<repo_dir\>/install/lib/OgmaNeo.lib -DOGMANEO_INCLUDE_DIR=\<repo_dir\>/install/include  

## Contributions

Refer to the OgmaNeo [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeo/blob/master/CONTRIBUTING.md) file for details about contributing to OgmaNeo, and the [Ogma Contributor Agreement](https://ogma.ai/wp-content/uploads/2016/09/OgmaContributorAgreement.pdf).

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [OGMANEO_LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/OGMANEO_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeo/blob/master/LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeo Copyright (c) 2016-2017 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.

