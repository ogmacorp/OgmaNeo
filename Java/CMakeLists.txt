# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(JOgmaNeo)
set(CMAKE_VERBOSE_MAKEFILE OFF)

#message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../install/CMake/")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(BITNESS 64)
else()
  set(BITNESS 32)
endif()
#message("Bitness: ${BITNESS}")

set(OGMANEO_PATH "${CMAKE_BINARY_DIR}/3rdparty")
include_directories(BEFORE "${OGMANEO_PATH}/include")
link_directories("${OGMANEO_PATH}/lib")


############################################################################
# Find OgmaNeo include and libs

if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR lib/CMake/OgmaNeo)
endif()

find_package(OgmaNeo CONFIG HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../install/${DEF_INSTALL_CMAKE_DIR}")

if ("${OGMANEO_INCLUDE_DIRS}" STREQUAL "OGMANEO_INCLUDE_DIRS-NOTFOUND" OR
    "${OGMANEO_INCLUDE_DIRS}" STREQUAL "")
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
    message(STATUS "Found existing OgmaNeo lib ${OGMANEO_LIBRARIES}")
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
# Find SWIG and setup building the Python bindings to OgmaNeo library

find_package(SWIG REQUIRED)
include(UseSWIG)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


############################################################################
# Find Java and JNI

find_package(Java)
find_package(JNI)

include(UseJava)

if (Java_FOUND)
    message(STATUS "Found Java version ${Java_VERSION}")
endif()

if (JNI_FOUND)
    message(STATUS "Found Java JNI in ${JNI_INCLUDE_DIRS}")
    message(STATUS "Found Java JNI lib ${JNI_LIBRARIES}")
endif()

include_directories(${JNI_INCLUDE_DIRS})


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_BINARY_DIR}/src/main/java/com/ogmacorp/ogmaneo")
set(CMAKE_SWIG_FLAGS -package com.ogmacorp.ogmaneo -outdir ${CMAKE_SWIG_OUTDIR} -I\"${OGMANEO_INCLUDE_DIRS}\")

set_source_files_properties(jogmaneo.i PROPERTIES CPLUSPLUS ON)

swig_add_module(JOgmaNeo java jogmaneo.i)
swig_link_libraries(JOgmaNeo ${JNI_LIBRARIES} ${OPENCL_LIBRARIES} ${OGMANEO_LIBRARIES})

if ("${OGMANEO_INCLUDE_DIRS}" STREQUAL "OGMANEO_INCLUDE_DIRS-NOTFOUND" OR
    "${OGMANEO_INCLUDE_DIRS}" STREQUAL "")
    add_dependencies(JOgmaNeo OgmaNeoDownload)
    set(SWIG_MODULE_JOgmaNeo_EXTRA_DEPS OgmaNeoDownload)
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/classes")
add_custom_command(
    TARGET JOgmaNeo POST_BUILD
    COMMAND cmake -E echo "Compiling Java files"
    COMMAND ${Java_JAVAC_EXECUTABLE} -d classes src/main/java/com/ogmacorp/ogmaneo/*.java
    COMMAND cmake -E echo "Creating Jar file"
    COMMAND ${Java_JAR_EXECUTABLE} cfM JOgmaNeo.jar -C classes .
    COMMAND cmake -E echo "Copying OgmaNeo library"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:JOgmaNeo> "${CMAKE_BINARY_DIR}"
)

