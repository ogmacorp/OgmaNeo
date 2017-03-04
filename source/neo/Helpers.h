// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "system/ComputeSystem.h"
#include "system/ComputeProgram.h"
#include "schemas/Helpers_generated.h"

#include <random>
#include <assert.h>

#ifdef _DEBUG
#define CL_CHECK(a) \
{\
	cl_int clCheckResult = (a);\
	if (clCheckResult != CL_SUCCESS)\
		std::cerr << ogmaneo::clErrorString(clCheckResult) << std::endl;\
}
#else
#define CL_CHECK(a) (a)
#endif

namespace ogmaneo {
#ifdef _DEBUG
    /*!
    \brief Output to cerr (stderr) a string form of an OpenCL error code
    \param[in] error OpenCL error code.
    */
    const char *clErrorString(cl_int error);
#endif

    /*!
    \brief Buffer types (can be used as indices)
    */
    enum BufferType {
        _front = 0, _back = 1
    };

    //!@{
    /*!
    \brief Double buffer types
    */
    typedef std::array<cl::Image2D, 2> DoubleBuffer2D;
    typedef std::array<cl::Image3D, 2> DoubleBuffer3D;
    //!@}

    //!@{
    /*!
    \brief Double buffer creation helpers
    */
    DoubleBuffer2D createDoubleBuffer2D(ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType);
    DoubleBuffer3D createDoubleBuffer3D(ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType);
    //!@}

    //!@{
    /*!
    \brief Double buffer initialization helpers
    */
    void randomUniform(cl::Image2D &image2D, ComputeSystem &cs, cl::Kernel &randomUniform2DKernel, cl_int2 size, cl_float4 lowerBounds, cl_float4 upperBounds, cl_float4 mask, cl_float4 fillConstants, std::mt19937 &rng);
    void randomUniform(cl::Image3D &image3D, ComputeSystem &cs, cl::Kernel &randomUniform3DKernel, cl_int3 size, cl_float4 lowerBounds, cl_float4 upperBounds, cl_float4 mask, cl_float4 fillConstants, std::mt19937 &rng);
    //!@}

    //!@{
    /*!
    \brief Image and Double buffer serialization helpers
    */
    void load(cl::Image2D &img, const schemas::Image2D* fbImg, ComputeSystem &cs);
    void load(cl::Image3D &img, const schemas::Image3D* fbImg, ComputeSystem &cs);
    flatbuffers::Offset<schemas::Image2D> save(cl::Image2D &img, flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
    flatbuffers::Offset<schemas::Image3D> save(cl::Image3D &img, flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);

    void load(DoubleBuffer2D &db, const schemas::DoubleBuffer2D* fbDB, ComputeSystem &cs);
    void load(DoubleBuffer3D &db, const schemas::DoubleBuffer3D* fbDB, ComputeSystem &cs);
    flatbuffers::Offset<schemas::DoubleBuffer2D> save(DoubleBuffer2D &db, flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
    flatbuffers::Offset<schemas::DoubleBuffer3D> save(DoubleBuffer3D &db, flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
    //!@}
}