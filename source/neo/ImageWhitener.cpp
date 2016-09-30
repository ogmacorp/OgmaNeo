// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageWhitener.h"
#include "Helpers.h"

using namespace ogmaneo;

void ImageWhitener::create(ComputeSystem &cs, ComputeProgram &program, cl_int2 imageSize, cl_int imageFormat, cl_int imageType) {
    _imageSize = imageSize;

    _result = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(imageFormat, imageType), imageSize.x, imageSize.y);

    _whitenKernel = cl::Kernel(program.getProgram(), "whiten");
}

void ImageWhitener::filter(ComputeSystem &cs, const cl::Image2D &input, cl_int kernelRadius, cl_float intensity) {
    int argIndex = 0;

    _whitenKernel.setArg(argIndex++, input);
    _whitenKernel.setArg(argIndex++, _result);
    _whitenKernel.setArg(argIndex++, _imageSize);
    _whitenKernel.setArg(argIndex++, kernelRadius);
    _whitenKernel.setArg(argIndex++, intensity);

    cs.getQueue().enqueueNDRangeKernel(_whitenKernel, cl::NullRange, cl::NDRange(_imageSize.x, _imageSize.y));
}

void ImageWhitener::load(const schemas::ImageWhitener* fbImageWhitener, ComputeSystem &cs, ComputeProgram& prog) {
    const schemas::Image2D *fbImage = fbImageWhitener->_result();

    // Loading into an uninitialized ImageWhitener?
    if (_imageSize.x == 0 || _imageSize.y == 0) {
        _imageSize.x = fbImageWhitener->_size()->x();
        _imageSize.y = fbImageWhitener->_size()->y();

        create(cs, prog, _imageSize,
            fbImage->format()->image_channel_order(),
            fbImage->format()->image_channel_data_type());
    }
    else {
        // Check we're loading into the same size ImageWhitener
        assert(_imageSize.x == fbImageWhitener->_size()->x());
        assert(_imageSize.y == fbImageWhitener->_size()->y());
    }

    ogmaneo::load(_result, fbImage, cs);
}

flatbuffers::Offset<schemas::ImageWhitener> ImageWhitener::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::int2 imageSize(_imageSize.x, _imageSize.y);

    return schemas::CreateImageWhitener(builder, &imageSize, ogmaneo::save(_result, builder, cs));
}
