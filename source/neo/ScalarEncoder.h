// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include <vector>

namespace ogmaneo {
    /*!
    \brief Simple RBF-based scalar encoder
    Encodes a set of scalars into an SDR. Optional learning included. CPU-only.
    */
    class OGMA_API ScalarEncoder {
    private:
        //!@{
        /*!
        \brief Weights are a 2D matrix
        */
        std::vector<float> _weightsEncode;
        std::vector<float> _weightsDecode;
        //!@}

        //!@{
        /*!
        \brief Encoder and decoder results
        */
        std::vector<float> _encoderOutputs;
        std::vector<float> _decoderOutputs;
        //!@}

        /*!
        \brief Bias values
        */
        std::vector<float> _biases;

    public:
        /*!
        \brief Randomly initialize the scalar encoder
        \param numInputs to the encoder.
        \param numOutputs output SDR size to decode.
        \param initMinWeight is the minimum value for weight initialization.
        \param initMaxWeight is the maximum value for weight initialization.
        \param seed rng seed, same seed gives same scalar encoder results.
        */
        void createRandom(int numInputs, int numOutputs, float initMinWeight, float initMaxWeight, int seed);

        /*!
        \brief Perform encoding
        \param inputs the inputs to be encoded.
        \param activeRatio % active units.
        \param alpha learning rate for weights (often 0 for this encoder).
        \param beta learning rate for biases (often 0 for this encoder).
        */
        void encode(const std::vector<float> &inputs, float activeRatio, float alpha, float beta);

        /*!
        \brief Perform decoding
        \param outputs SDR to decode.
        */
        void decode(const std::vector<float> &outputs);

        /*!
        \brief Get resulting encoding
        */
        std::vector<float> &getEncoderOutputs() {
            return _encoderOutputs;
        }

        /*!
        \brief Get resulting decoding
        */
        std::vector<float> &getDecoderOutputs() {
            return _decoderOutputs;
        }
    };
}