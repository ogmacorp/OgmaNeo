# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(PyOgmaNeo)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/OgmaNeo)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../install/${DEF_INSTALL_CMAKE_DIR}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()
message("Bitness: ${BITNESS}")

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(PYTHON_VERSION)
    message("Python target version: ${PYTHON_VERSION}")            
endif()

set(OGMANEO_PATH "${CMAKE_BINARY_DIR}/3rdparty")
include_directories(BEFORE "${OGMANEO_PATH}/include")
link_directories("${OGMANEO_PATH}/lib")


############################################################################
# Find OgmaNeo include and libs

find_package(OgmaNeo CONFIG HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../install//${DEF_INSTALL_CMAKE_DIR}")

if ("${OGMANEO_INCLUDE_DIR}" STREQUAL "OGMANEO_INCLUDE_DIR-NOTFOUND" OR
    "${OGMANEO_INCLUDE_DIR}" STREQUAL "")
    message(STATUS "-- OgmaNeo will be downloaded and built from source")

    if(WIN32 OR APPLE)
        set(OGMANEO_SHARED_LIBS False)
    else()
        set(OGMANEO_SHARED_LIBS True)
    endif()

    include(ExternalProject)
    ExternalProject_Add(OgmaNeoDownload
        PREFIX OgmaNeo
        GIT_REPOSITORY https://github.com/ogmacorp/ogmaneo.git
        TIMEOUT 30
        CMAKE_ARGS
            "-DBUILD_SHARED_LIBS=${OGMANEO_SHARED_LIBS}"
            "-DCMAKE_INSTALL_PREFIX=${OGMANEO_PATH}"
    )
    set(OGMANEO_INCLUDE_DIRS "${OGMANEO_PATH}/include/ogmaneo")

    if(WIN32 OR APPLE)
        set(OGMANEO_LIBRARIES "${OGMANEO_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}OgmaNeo${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(OGMANEO_LIBRARIES "${OGMANEO_PATH}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}OgmaNeo${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()
else()
    message(STATUS "Found existing OgmaNeo in ${OGMANEO_INCLUDE_DIRS}")
    #message(STATUS "Found existing OgmaNeo lib ${OGMANEO_LIBRARIES}")
endif()

include_directories(${OGMANEO_INCLUDE_DIRS})


############################################################################
# Find OpenCL include and libs

#find_package(OpenCL REQUIRED)
#include_directories(${OpenCL_INCLUDE_DIRS})

find_path(OPENCL_INCLUDE_DIRS 
    NAMES OpenCL/cl.h CL/cl.h
    HINTS
        $ENV{AMDAPPSDKROOT}/include/
        $ENV{INTELOCLSDKROOT}/include/
        $ENV{CUDA_PATH}/include/
        $ENV{OPENCL_ROOT}/include/
    PATHS
        /usr/include /usr/local/include/
)

if(BITNESS EQUAL 64)
    find_library(OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86_64 x64
        PATHS /usr/lib64 /usr/lib /usr/local/lib
    )
elseif(BITNESS EQUAL 32)
    find_library(OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86 Win32
        PATHS /usr/lib32 /usr/lib /usr/local/lib
    )
endif()

# Fall-back to find a POCL install
if(OPENCL_LIBRARIES STREQUAL "")
    find_library(OPENCL_LIBRARIES
        NAMES pocl
        HINTS $HOME/.local/pocl/lib /usr/lib64 /usr/lib /usr/local/lib
    )
endif()

if(OPENCL_INCLUDE_DIRS STREQUAL "" OR OPENCL_LIBRARIES STREQUAL "")
    message(FATAL_ERROR "Could not locate OpenCL include & libs")
else()
    message(STATUS "Found existing OpenCL in ${OPENCL_INCLUDE_DIRS}")
    message(STATUS "Found existing OpenCL lib ${OPENCL_LIBRARIES}")
endif()

include_directories(${OPENCL_INCLUDE_DIRS})


############################################################################
# Find Khronos cl2.hpp include file

find_file(CL2_HPP_NAME
    NAMES OpenCL/cl2.hpp CL/cl2.hpp
    HINTS
        $ENV{AMDAPPSDKROOT}/include/
        $ENV{INTELOCLSDKROOT}/include/
        $ENV{CUDA_PATH}/include/
        $ENV{OPENCL_ROOT}/include/
    PATHS
        /usr/include /usr/local/include/
)

if ("${CL2_HPP_NAME}" STREQUAL "CL2_HPP_NAME-NOTFOUND")
    message(STATUS "CL2 HPP not found!")

    # The file DOWNLOAD using a https URL require that CMake contains SSL support
    # By default CMake doesn't have SSL enabled. Only Mac OSX has this enabled,
    # so restict the following cl2.hpp download to only Apple devices.

    if(APPLE)
        message(STATUS "Downloading cl2.hpp v2.0.10 into ${OGMANEO_PATH}/include")
        file(DOWNLOAD
            https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp
            ${OGMANEO_PATH}/include/OpenCL/cl2.hpp
            STATUS cl2_download_status
            LOG cl2_download_log)

        list(GET cl2_download_status 0 status_code)
        list(GET cl2_download_status 1 status_string)

        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "Error: downloading cl2.hpp (code: ${status_code} - ${status_string} log: ${cl2_download_log}")
        endif()
    endif()
else()
    # Extract the directory
    get_filename_component(CL2_HPP_DIR "${CL2_HPP_NAME}" DIRECTORY)
    message(STATUS "Found existing CL2.hpp in ${CL2_HPP_DIR}")

    include_directories(SYSTEM "${CL2_HPP_DIR}/..")
endif()


############################################################################
# Find SWIG and setup building the Python bindings to OgmaNeo library

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


############################################################################
# Find Python package

# Are we dealing with Anaconda?
set(CONDA_COMMAND conda info --root)
execute_process(
    COMMAND ${CONDA_COMMAND}
    RESULT_VARIABLE _CONDA_RESULT
    OUTPUT_VARIABLE _CONDA_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(${_CONDA_RESULT} STREQUAL "0")
    string(TOLOWER ${_CONDA_OUTPUT} _CONDA_OUTPUT)
    string(REPLACE "\\" "/" _CONDA_OUTPUT ${_CONDA_OUTPUT})
    string(REGEX MATCH "anaconda" ANACONDA_PYTHON_FOUND "${_CONDA_OUTPUT}")
endif()

if(ANACONDA_PYTHON_FOUND)
    set(ANACONDA_PYTHON_DIR ${_CONDA_OUTPUT})
    message("Found anaconda in ${ANACONDA_PYTHON_DIR}")

    set(_PYTHON_COMMAND python --version)
    execute_process(
        COMMAND ${_PYTHON_COMMAND}
        ERROR_VARIABLE _PYTHON_ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "Python ([0-9]+)[.]([0-9]+)[.]([0-9]+)" _PYTHON_VERSION_FOUND "${_PYTHON_ERROR}")
    set(_PYTHON_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(_PYTHON_VERSION_MINOR ${CMAKE_MATCH_2})
    set(_PYTHON_VERSION_PATCH ${CMAKE_MATCH_3})

    if(WIN32)
        set(ANACONDA_PYTHON_VERSION ${_PYTHON_VERSION_MAJOR}${_PYTHON_VERSION_MINOR})
        set(_PYTHON_LIB "libs/python${ANACONDA_PYTHON_VERSION}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(ANACONDA_PYTHON_VERSION ${_PYTHON_VERSION_MAJOR}.${_PYTHON_VERSION_MINOR})
        if(${_PYTHON_VERSION_MAJOR} MATCHES 2)
            set(_PYTHON_EXTENSION "")
        else()
            set(_PYTHON_EXTENSION "m")
        endif()
        set(_PYTHON_ID "python${ANACONDA_PYTHON_VERSION}${_PYTHON_EXTENSION}")
        set(_PYTHON_LIB "lib/${CMAKE_SHARED_LIBRARY_PREFIX}python${ANACONDA_PYTHON_VERSION}${_PYTHON_EXTENSION}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

    if(NOT DEFINED ENV{CONDA_DEFAULT_ENV})
        set(PYTHON_INCLUDE_DIR "${ANACONDA_PYTHON_DIR}/include/${_PYTHON_ID}" CACHE INTERNAL "")
        set(PYTHON_LIBRARY "${ANACONDA_PYTHON_DIR}/${_PYTHON_LIB}" CACHE INTERNAL "")
    else()
        set(_CONDA_DEFAULT_ENV $ENV{CONDA_DEFAULT_ENV})
        string(REPLACE "\\" "/" _CONDA_DEFAULT_ENV ${_CONDA_DEFAULT_ENV})
        set(PYTHON_INCLUDE_DIR "${ANACONDA_PYTHON_DIR}/envs/${_CONDA_DEFAULT_ENV}/include/${_PYTHON_ID}" CACHE INTERNAL "")
        set(PYTHON_LIBRARY "${ANACONDA_PYTHON_DIR}/envs/${_CONDA_DEFAULT_ENV}/${_PYTHON_LIB}" CACHE INTERNAL "")
    endif()

    message("PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}")
    message("PYTHON_LIBRARY = ${PYTHON_LIBRARY}")

    set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
    set(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")
    set(FOUND_PYTHONLIBS TRUE)
else()
    if(DEFINED ENV{TRAVIS} AND DEFINED ENV{TRAVIS_PYTHON_VERSION})
        message(STATUS "Finding PythonLibs $ENV{TRAVIS_PYTHON_VERSION}")
        if("$ENV{TRAVIS_PYTHON_VERSION}" STREQUAL "2.7")
            find_package(PythonLibs $ENV{TRAVIS_PYTHON_VERSION})
        else()
            if(NOT PYTHON_VERSION)
                find_package(PythonLibs)
            else()
                find_package(PythonLibs ${PYTHON_VERSION})            
            endif()
        endif()
    else()
        if(NOT PYTHON_VERSION)
            find_package(PythonLibs)
        else()
            find_package(PythonLibs ${PYTHON_VERSION})            
        endif()
    endif()
endif()

include_directories(${PYTHON_INCLUDE_DIR})


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_BINARY_DIR}")
set(CMAKE_SWIG_FLAGS "-I\"${PYTHON_INCLUDE_DIR}\" -I\"${OPENCL_INCLUDE_DIRS}\" -I\"${OGMANEO_INCLUDE_DIRS}\"")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

set_source_files_properties(pyogmaneo.i PROPERTIES CPLUSPLUS ON)

swig_add_module(ogmaneo python pyogmaneo.i)
swig_link_libraries(ogmaneo ${PYTHON_LIBRARIES} ${OPENCL_LIBRARIES} ${OGMANEO_LIBRARIES})

if ("${OGMANEO_INCLUDE_DIR}" STREQUAL "OGMANEO_INCLUDE_DIR-NOTFOUND" OR
    "${OGMANEO_INCLUDE_DIR}" STREQUAL "")
    add_dependencies(_ogmaneo OgmaNeoDownload)
    set_property(TARGET _ogmaneo PROPERTY DEPENDS OgmaNeoDownload)
    set(SWIG_MODULE_ogmaneo_EXTRA_DEPS OgmaNeoDownload)
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_SWIG_OUTDIR})
install(TARGETS _ogmaneo DESTINATION ${CMAKE_SWIG_OUTDIR})
install(FILES ${CMAKE_BINARY_DIR}/ogmaneo.py DESTINATION ${CMAKE_SWIG_OUTDIR})

