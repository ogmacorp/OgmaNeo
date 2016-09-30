// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "OgmaNeo.h"
#include "schemas/ImageWhitener_generated.h"

namespace ogmaneo {
    /*!
    \brief Image whitener.
    Applies local whitening transformation to input.
    */
    class OGMA_API ImageWhitener {
    private:
        /*!
        \brief Kernels
        */
        cl::Kernel _whitenKernel;

        /*!
        \brief Resulting whitened image
        */
        cl::Image2D _result;

        /*!
        \brief Size of the whitened image
        */
        cl_int2 _imageSize;

    public:
        /*!
        \brief Create the image whitener.
        Requires the image size and format, and ComputeProgram loaded with the extra kernel code (see ComputeProgram::loadExtraKernel).
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the extra kernel code.
        \param imageSize size of the source image (2D).
        \param imageFormat format of the image, e.g. CL_R, CL_RG, CL_RGB, CL_RGBA.
        \param imageType type of the image, e.g. CL_FLOAT, CL_UNORM_INT8.
        */
        void create(ComputeSystem &cs, ComputeProgram &program, cl_int2 imageSize, cl_int imageFormat, cl_int imageType);

        /*!
        \brief Filter (whiten) an image with a kernel radius
        \param cs is the ComputeSystem.
        \param input is the OpenCL 2D image to be whitened.
        \param kernelRadius local radius of examined pixels.
        \param intensity the strength of the whitening.
        */
        void filter(ComputeSystem &cs, const cl::Image2D &input, cl_int kernelRadius, cl_float intensity = 1024.0f);

        /*!
        \brief Return filtered image result
        */
        const cl::Image2D &getResult() const {
            return _result;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::ImageWhitener* fbImgWhitener, ComputeSystem &cs, ComputeProgram& prog);
        flatbuffers::Offset<schemas::ImageWhitener> save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs);
        //!@}
    };
}