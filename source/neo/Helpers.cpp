// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Helpers.h"

using namespace ogmaneo;

DoubleBuffer2D ogmaneo::createDoubleBuffer2D(ComputeSystem &cs, cl_int2 size, cl_channel_order channelOrder, cl_channel_type channelType) {
    DoubleBuffer2D db;

    db[_front] = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y);
    db[_back] = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y);

    return db;
}

DoubleBuffer3D ogmaneo::createDoubleBuffer3D(ComputeSystem &cs, cl_int3 size, cl_channel_order channelOrder, cl_channel_type channelType) {
    DoubleBuffer3D db;

    db[_front] = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y, size.z);
    db[_back] = cl::Image3D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(channelOrder, channelType), size.x, size.y, size.z);

    return db;
}

void ogmaneo::randomUniform(cl::Image2D &image2D, ComputeSystem &cs, cl::Kernel &randomUniform2DKernel, cl_int2 size, cl_float4 lowerBounds, cl_float4 upperBounds, cl_float4 mask, cl_float4 fillConstants, std::mt19937 &rng) {
    int argIndex = 0;

    std::uniform_int_distribution<int> seedDist(0, 999);

    cl_uint2 seed = { (cl_uint)seedDist(rng), (cl_uint)seedDist(rng) };

    randomUniform2DKernel.setArg(argIndex++, image2D);
    randomUniform2DKernel.setArg(argIndex++, seed);
    randomUniform2DKernel.setArg(argIndex++, lowerBounds);
    randomUniform2DKernel.setArg(argIndex++, upperBounds);
    randomUniform2DKernel.setArg(argIndex++, mask);
    randomUniform2DKernel.setArg(argIndex++, fillConstants);

    cs.getQueue().enqueueNDRangeKernel(randomUniform2DKernel, cl::NullRange, cl::NDRange(size.x, size.y));
}

void ogmaneo::randomUniform(cl::Image3D &image3D, ComputeSystem &cs, cl::Kernel &randomUniform3DKernel, cl_int3 size, cl_float4 lowerBounds, cl_float4 upperBounds, cl_float4 mask, cl_float4 fillConstants, std::mt19937 &rng) {
    int argIndex = 0;

    std::uniform_int_distribution<int> seedDist(0, 999);

    cl_uint2 seed = { (cl_uint)seedDist(rng), (cl_uint)seedDist(rng) };

    randomUniform3DKernel.setArg(argIndex++, image3D);
    randomUniform3DKernel.setArg(argIndex++, seed);
    randomUniform3DKernel.setArg(argIndex++, lowerBounds);
    randomUniform3DKernel.setArg(argIndex++, upperBounds);
    randomUniform3DKernel.setArg(argIndex++, mask);
    randomUniform3DKernel.setArg(argIndex++, fillConstants);

    cs.getQueue().enqueueNDRangeKernel(randomUniform3DKernel, cl::NullRange, cl::NDRange(size.x, size.y, size.z));
}

void ogmaneo::load(cl::Image2D &img, const schemas::Image2D* fbImg, ComputeSystem &cs) {
    uint32_t width = (uint32_t)img.getImageInfo<CL_IMAGE_WIDTH>();
    uint32_t height = (uint32_t)img.getImageInfo<CL_IMAGE_HEIGHT>();
    uint32_t elementSize = (uint32_t)img.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();

    assert(width == fbImg->width());
    assert(height == fbImg->height());
    assert(elementSize == fbImg->elementSize());

    schemas::PixelData pixelType = fbImg->pixels_type();
    switch (fbImg->pixels_type())
    {
    case schemas::PixelData::PixelData_FloatArray:
    {
        const schemas::FloatArray* fbFloatArray =
            reinterpret_cast<const schemas::FloatArray*>(fbImg->pixels());

        cs.getQueue().enqueueWriteImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, 1 }, 0, 0, static_cast<void*>(const_cast<float*>(fbFloatArray->data()->data())));
        cs.getQueue().finish();
        break;
    }
    case schemas::PixelData::PixelData_ByteArray:
    {
        const schemas::ByteArray* fbByteArray =
            reinterpret_cast<const schemas::ByteArray*>(fbImg->pixels());

        cs.getQueue().enqueueWriteImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, 1 }, 0, 0, static_cast<void*>(const_cast<uint8_t*>(fbByteArray->data()->data())));
        cs.getQueue().finish();
        break;
    }
    default:
        assert(0);
        break;
    }
}

void ogmaneo::load(cl::Image3D &img, const schemas::Image3D* fbImg, ComputeSystem &cs) {
    uint32_t width = (uint32_t)img.getImageInfo<CL_IMAGE_WIDTH>();
    uint32_t height = (uint32_t)img.getImageInfo<CL_IMAGE_HEIGHT>();
    uint32_t depth = (uint32_t)img.getImageInfo<CL_IMAGE_DEPTH>();
    uint32_t elementSize = (uint32_t)img.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();

    assert(width == fbImg->width());
    assert(height == fbImg->height());
    assert(depth == fbImg->depth());
    assert(elementSize == fbImg->elementSize());

    schemas::PixelData pixelType = fbImg->pixels_type();
    switch (fbImg->pixels_type())
    {
    case schemas::PixelData::PixelData_FloatArray:
    {
        const schemas::FloatArray* fbFloatArray =
            reinterpret_cast<const schemas::FloatArray*>(fbImg->pixels());

        cs.getQueue().enqueueWriteImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, depth }, 0, 0, static_cast<void*>(const_cast<float*>(fbFloatArray->data()->data())));
        cs.getQueue().finish();
        break;
    }
    case schemas::PixelData::PixelData_ByteArray:
    {
        const schemas::ByteArray* fbByteArray =
            reinterpret_cast<const schemas::ByteArray*>(fbImg->pixels());

        cs.getQueue().enqueueWriteImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, depth }, 0, 0, static_cast<void*>(const_cast<uint8_t*>(fbByteArray->data()->data())));
        cs.getQueue().finish();
        break;
    }
    default:
        assert(0);
        break;
    }
}

flatbuffers::Offset<schemas::Image2D> ogmaneo::save(cl::Image2D &img, flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    uint32_t width = (uint32_t)img.getImageInfo<CL_IMAGE_WIDTH>();
    uint32_t height = (uint32_t)img.getImageInfo<CL_IMAGE_HEIGHT>();
    uint32_t elementSize = (uint32_t)img.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();

    cl_channel_order channelOrder = img.getImageInfo<CL_IMAGE_FORMAT>().image_channel_order;
    cl_channel_type channelType = img.getImageInfo<CL_IMAGE_FORMAT>().image_channel_data_type;

    schemas::ImageFormat format(
        static_cast<schemas::ChannelOrder>(channelOrder),
        static_cast<schemas::ChannelDataType>(channelType)
    );

    flatbuffers::Offset<schemas::Image2D> ret;

    switch (channelType) {
    case CL_FLOAT:
    {
        std::vector<float> pixels(width * height * (elementSize / sizeof(float)), 0.0f);
        cs.getQueue().enqueueReadImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, 1 }, 0, 0, pixels.data());
        cs.getQueue().finish();

        flatbuffers::Offset<flatbuffers::Vector<float>> floatVector = builder.CreateVector(pixels.data(), pixels.size());
        flatbuffers::Offset<schemas::FloatArray> floatArray = schemas::CreateFloatArray(builder, floatVector);
        ret = schemas::CreateImage2D(builder,
            &format, width, height, elementSize, schemas::PixelData_FloatArray, floatArray.Union());
        break;
    }
    case CL_UNSIGNED_INT8:
    case CL_SIGNED_INT8:
    {
        std::vector<unsigned char> pixels(width * height * (elementSize / sizeof(unsigned char)), 0);
        cs.getQueue().enqueueReadImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, 1 }, 0, 0, pixels.data());
        cs.getQueue().finish();

        flatbuffers::Offset<flatbuffers::Vector<unsigned char>> byteVector = builder.CreateVector(pixels.data(), pixels.size());
        flatbuffers::Offset<schemas::ByteArray> byteArray = schemas::CreateByteArray(builder, byteVector);
        ret = schemas::CreateImage2D(builder,
            &format, width, height, elementSize, schemas::PixelData_ByteArray, byteArray.Union());
        break;
    }
    default:
        assert(0);
        break;
    }

    return ret;
}

flatbuffers::Offset<schemas::Image3D> ogmaneo::save(cl::Image3D &img, flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    uint32_t width = (uint32_t)img.getImageInfo<CL_IMAGE_WIDTH>();
    uint32_t height = (uint32_t)img.getImageInfo<CL_IMAGE_HEIGHT>();
    uint32_t depth = (uint32_t)img.getImageInfo<CL_IMAGE_DEPTH>();
    uint32_t elementSize = (uint32_t)img.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();

    cl_channel_order channelOrder = img.getImageInfo<CL_IMAGE_FORMAT>().image_channel_order;
    cl_channel_type channelType = img.getImageInfo<CL_IMAGE_FORMAT>().image_channel_data_type;

    schemas::ImageFormat format(
        static_cast<schemas::ChannelOrder>(channelOrder),
        static_cast<schemas::ChannelDataType>(channelType)
    );

    flatbuffers::Offset<schemas::Image3D> ret;

    switch (channelType) {
    case CL_FLOAT:
    {
        std::vector<float> pixels(width * height * depth * (elementSize / sizeof(float)), 0.0f);
        cs.getQueue().enqueueReadImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, depth }, 0, 0, pixels.data());
        cs.getQueue().finish();

        flatbuffers::Offset<flatbuffers::Vector<float>> floatVector = builder.CreateVector(pixels.data(), pixels.size());
        flatbuffers::Offset<schemas::FloatArray> floatArray = schemas::CreateFloatArray(builder, floatVector);
        ret = schemas::CreateImage3D(builder,
            &format, width, height, depth, elementSize, schemas::PixelData_FloatArray, floatArray.Union());
        break;
    }
    case CL_UNSIGNED_INT8:
    case CL_SIGNED_INT8:
    {
        std::vector<unsigned char> pixels(width * height * depth * (elementSize / sizeof(unsigned char)), 0);
        cs.getQueue().enqueueReadImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, depth }, 0, 0, pixels.data());
        cs.getQueue().finish();

        flatbuffers::Offset<flatbuffers::Vector<unsigned char>> byteVector = builder.CreateVector(pixels.data(), pixels.size());
        flatbuffers::Offset<schemas::ByteArray> byteArray = schemas::CreateByteArray(builder, byteVector);
        ret = schemas::CreateImage3D(builder,
            &format, width, height, depth, elementSize, schemas::PixelData_ByteArray, byteArray.Union());
        break;
    }
    default:
        assert(0);
        break;
    }

    return ret;
}

void ogmaneo::load(DoubleBuffer2D &db, const schemas::DoubleBuffer2D* fbDB, ComputeSystem &cs) {
    if (db[_front].get() == nullptr || db[_back].get() == nullptr)
        return;

    ogmaneo::load(db[_front], fbDB->_front(), cs);
    ogmaneo::load(db[_back], fbDB->_back(), cs);
}

void ogmaneo::load(DoubleBuffer3D &db, const schemas::DoubleBuffer3D* fbDB, ComputeSystem &cs) {
    if (db[_front].get() == nullptr || db[_back].get() == nullptr)
        return;

    ogmaneo::load(db[_front], fbDB->_front(), cs);
    ogmaneo::load(db[_back], fbDB->_back(), cs);
}

flatbuffers::Offset<schemas::DoubleBuffer2D> ogmaneo::save(DoubleBuffer2D &db, flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    if (db[_front].get() == nullptr || db[_back].get() == nullptr)
        return schemas::CreateDoubleBuffer2D(builder, 0, 0);

    return schemas::CreateDoubleBuffer2D(builder,
        ogmaneo::save(db[_front], builder, cs),
        ogmaneo::save(db[_back], builder, cs)
    );
}

flatbuffers::Offset<schemas::DoubleBuffer3D> ogmaneo::save(DoubleBuffer3D &db, flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    if (db[_front].get() == nullptr || db[_back].get() == nullptr)
        return schemas::CreateDoubleBuffer3D(builder, 0, 0);

    return schemas::CreateDoubleBuffer3D(builder,
        ogmaneo::save(db[_front], builder, cs),
        ogmaneo::save(db[_back], builder, cs)
    );
}
