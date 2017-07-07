# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(OgmaNeo)
set(CMAKE_VERBOSE_MAKEFILE OFF)

set(OGMANEO_MAJOR_VERSION 1)
set(OGMANEO_MINOR_VERSION 4)
set(OGMANEO_PATCH_VERSION 0)
set(OGMANEO_VERSION
  ${OGMANEO_MAJOR_VERSION}.${OGMANEO_MINOR_VERSION}.${OGMANEO_PATCH_VERSION})
 
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITNESS 64)
else()
    set(BITNESS 32)
endif()
message(STATUS "Bitness: ${BITNESS}")

option(BUILD_SHARED_LIBS OFF)
message(STATUS "Shared libs: ${BUILD_SHARED_LIBS}")


include(ExternalProject)

set(3RDPARTY_PATH "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")
include_directories(BEFORE "${3RDPARTY_PATH}/include")
link_directories("${3RDPARTY_PATH}/lib")


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
        message(STATUS "Downloading cl2.hpp v2.0.10 into ${3RDPARTY_PATH}/include")
        file(DOWNLOAD
            https://github.com/KhronosGroup/OpenCL-CLHPP/releases/download/v2.0.10/cl2.hpp
            ${3RDPARTY_PATH}/include/OpenCL/cl2.hpp
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
# Find flatbuffers and build schemas

find_package(FlatBuffers)

if(FLATBUFFERS_FOUND)
    message(STATUS "Found existing FlatBuffers in ${FLATBUFFERS_INCLUDE_DIRS}")
    include_directories(${FLATBUFFERS_INCLUDE_DIRS})
else()
    message(STATUS "-- FlatBuffers will be downloaded and built from source")
    ExternalProject_Add(
        FlatBuffers
        PREFIX FlatBuffers
        GIT_REPOSITORY "https://github.com/google/flatbuffers.git"
        CMAKE_ARGS
            "-DFLATBUFFERS_BUILD_TESTS=OFF"
            "-DCMAKE_INSTALL_PREFIX=${3RDPARTY_PATH}"
    )
    set(FLATBUFFERS_INCLUDE_DIRS "${3RDPARTY_PATH}/include")
    set(FLATBUFFERS_FLATC_EXECUTABLE "${3RDPARTY_PATH}/bin/flatc${CMAKE_EXECUTABLE_SUFFIX}")
endif()

file(GLOB_RECURSE FBS_FILES "source/*.fbs")
#message(STATUS "Found fbs files - ${FBS_FILES}")

build_flatbuffers(
    "${FBS_FILES}"
    "${PROJECT_SOURCE_DIR}/source/schemas"
    OgmaNeoSchemas
    "" # No additional_dependencies
    "${PROJECT_SOURCE_DIR}/source/schemas"
    "" # No binary_schemas_dir
    "" # No copy_text_schemas_dir
)

set_property(TARGET OgmaNeoSchemas APPEND PROPERTY SOURCES ${FBS_FILES})
if(NOT FLATBUFFERS_FOUND)
    add_dependencies(OgmaNeoSchemas FlatBuffers)
endif()


############################################################################
# Find project include and sources

# Make sure we know where the OCLpp executable is
set(EXECUTABLE_OUTPUT_PATH "${OgmaNeo_BINARY_DIR}/bin" CACHE INTERNAL "")
set(LIBRARY_OUTPUT_PATH "${OgmaNeo_BINARY_DIR}/bin" CACHE INTERNAL "")

# Create the OpenCl to H preprocessor executable
add_executable(OgmaOCLtoHpp utils/OCLpp.cpp)

# Variable to store output files
set(outFiles)

# Find all OpenCL kernel files (note: not recursive)
file(GLOB KERNEL_SRC
          RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/resources/kernels"
          "${CMAKE_CURRENT_SOURCE_DIR}/resources/kernels/*.cl")
message(STATUS "Found CL files - ${KERNEL_SRC}")

set(infileCommon "${CMAKE_CURRENT_SOURCE_DIR}/resources/neoKernelsCommon.cl")
    
# Add a custom command to convert the OpenCL files into C++ header
# files containing an appropriately name string const block
foreach(ocl_filename ${KERNEL_SRC})
	set(infile "${CMAKE_CURRENT_SOURCE_DIR}/resources/kernels/${ocl_filename}")
    #message(STATUS "Input file: ${infile}")

    # Extract the filename
    string(REGEX REPLACE "\\.[^.]*$" "" filename ${ocl_filename})
    set(outfile "${CMAKE_CURRENT_SOURCE_DIR}/source/kernels/${filename}.h")
    #message(STATUS "Output file: ${outfile}")

    # Does the output directory exist?
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/source/kernels")
        execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/source/kernels")
    endif()

    add_custom_command(
        OUTPUT "${outfile}"
	    COMMAND OgmaOCLtoHpp "${infileCommon}" "${infile}" "${outfile}" ${filename}
        DEPENDS "${infileCommon}" "${infile}" OgmaOCLtoHpp
        VERBATIM)

    set(outFiles ${outFiles} "${outfile}")
endforeach()
#message(STATUS "Processed files: ${outFiles}")

# Setup a target to drive the conversion
add_custom_target(OgmaOCLtoH ALL DEPENDS ${outFiles})


# Find list of library source code
file(GLOB_RECURSE LIBRARY_SRC
    "source/neo/*.h"
    "source/neo/*.cpp"
    "source/system/*.h"
    "source/system/*.cpp"
)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/source")

# Build the main OgmaNeo library, as static or shared
add_library(OgmaNeo ${LIBRARY_SRC})

if(BUILD_SHARED_LIBS)
    add_definitions(-DOGMA_DLL)

#    Use CMake to generate SharedLib.h instead?
#    include (GenerateExportHeader)
#    generate_export_header(OgmaNeo
#                           BASE_NAME OgmaNeo
#                           EXPORT_MACRO_NAME OgmaNeo_EXPORT
#                           EXPORT_FILE_NAME OgmaNeo_Export.h
#                           STATIC_DEFINE OgmaNeo_BUILT_AS_STATIC
#    )
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Main library depends upon Schema compilation
# and OpenCL to H file generation
add_dependencies(OgmaNeo OgmaNeoSchemas OgmaOCLtoH)
target_link_libraries(OgmaNeo ${OPENCL_LIBRARIES})

set_property(TARGET OgmaNeo PROPERTY CXX_STANDARD 14)
set_property(TARGET OgmaNeo PROPERTY CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    if(BITNESS EQUAL 64)
        set_target_properties(OgmaNeo PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
    endif()
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include/ogmaneo CACHE PATH
  "Installation directory for header files")

message(STATUS "CMake install prefix = ${CMAKE_INSTALL_PREFIX}")

if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/OgmaNeo)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
 
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()
 
# Add all targets to the build-tree export set
export(TARGETS OgmaNeo
  FILE "${PROJECT_BINARY_DIR}/OgmaNeoTargets.cmake")
 
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE OgmaNeo)
 
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
   "${INSTALL_INCLUDE_DIR}")

set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(OgmaNeoConfig.cmake.in
  "${PROJECT_BINARY_DIR}/OgmaNeoConfig.cmake" @ONLY)

set(CONF_INCLUDE_DIRS "\${OGMANEO_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(OgmaNeoConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/OgmaNeoConfig.cmake" @ONLY)

configure_file(OgmaNeoConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/OgmaNeoConfigVersion.cmake" @ONLY)
 
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/OgmaNeoConfig.cmake"
  "${PROJECT_BINARY_DIR}/OgmaNeoConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}")
  
# Library install target
install(TARGETS OgmaNeo
        EXPORT OgmaNeoTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Headers, incl. schema headers, install targets
install(DIRECTORY "${PROJECT_SOURCE_DIR}/source/"
        EXPORT OgmaNeoTargets
        DESTINATION include/ogmaneo
        FILES_MATCHING PATTERN "*.h*")

# 3rd party headers install targets
install(DIRECTORY "${3RDPARTY_PATH}/include/"
        EXPORT OgmaNeoTargets
        DESTINATION include/ogmaneo
        FILES_MATCHING PATTERN "*.h*")
 
install(EXPORT OgmaNeoTargets DESTINATION "${INSTALL_CMAKE_DIR}")

# Uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/CMake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

