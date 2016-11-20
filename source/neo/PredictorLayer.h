// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "Helpers.h"
#include "SparseFeatures.h"
#include "schemas/PredictorLayer_generated.h"

namespace ogmaneo {
    /*!
    \brief Predictor layer.
    A 2D perceptron decoder (Predictor) layer.
    */
    class OGMA_API PredictorLayer {
    public:
        /*!
        \brief Layer descriptor
        */
        struct VisibleLayerDesc {
            //!@{
            /*!
            \brief Layer properties
            Input size, radius onto input, learning rate.
            */
            cl_int2 _size;

            cl_int _radius;

            cl_float _alpha;
            //!@}

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 16, 16 }),
                _radius(8),
                _alpha(0.04f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisiblePredictorLayerDesc* fbVisiblePredictorLayerDesc, ComputeSystem &cs);
            schemas::VisiblePredictorLayerDesc save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Layer
        */
        struct VisibleLayer {
            //!@{
            /*!
            \brief Layer parameters
            */
            DoubleBuffer2D _derivedInput;

            DoubleBuffer3D _weights;

            cl_float2 _hiddenToVisible;
            cl_float2 _visibleToHidden;

            cl_int2 _reverseRadii;
            //!@}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisiblePredictorLayer* fbVisiblePredictorLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::VisiblePredictorLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        /*!
        \brief Size of the prediction
        */
        cl_int2 _hiddenSize;

        /*!
        \brief Hidden stimulus summation temporary buffer
        */
        DoubleBuffer2D _hiddenSummationTemp;

        /*!
        \brief Predictions
        */
        DoubleBuffer2D _hiddenStates;

        /*!
        \brief
        Encoder corresponding to this decoder, if available (else nullptr)
        */
        std::shared_ptr<SparseFeatures> _inhibitSparseFeatures;

        //!@{
        /*!
        \brief Layers and descs
        */
        std::vector<VisibleLayer> _visibleLayers;
        std::vector<VisibleLayerDesc> _visibleLayerDescs;
        //!@}

        //!@{
        /*!
        \brief Additional kernels
        */
        cl::Kernel _deriveInputsKernel;
        cl::Kernel _stimulusKernel;
        cl::Kernel _learnPredWeightsKernel;
        //!@}

    public:
        /*!
        \brief Create a predictor layer with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param hiddenSize size of the predictions (output).
        \param visibleLayerDescs are descriptors for visible layers.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &plProgram,
            cl_int2 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            const std::shared_ptr<SparseFeatures> &inhibitSparseFeatures,
            cl_float2 initWeightRange, std::mt19937 &rng);

        /*!
        \brief Activate predictor (predict values)
        \param cs is the ComputeSystem.
        \param visibleStates the input layer states.
        \param threshold whether or not the output should be thresholded (binary).
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, std::mt19937 &rng);

        /*!
        \brief Learn predictor
        \param cs is the ComputeSystem.
        \param targets target values to update towards.
        \param visibleStatesPrev the input states of the !previous! timestep.
        */
        void learn(ComputeSystem &cs, const cl::Image2D &targets);

        /*!
        \brief Step end (buffer swap)
        \param cs is the ComputeSystem.
        */
        void stepEnd(ComputeSystem &cs);

        /*!
        \brief Clear memory (recurrent data)
        \param cs is the ComputeSystem.
        */
        void clearMemory(ComputeSystem &cs);

        /*!
        \brief Get number of layers
        */
        size_t getNumLayers() const {
            return _visibleLayers.size();
        }

        /*!
        \brief Get access to a layer
        */
        const VisibleLayer &getLayer(int index) const {
            return _visibleLayers[index];
        }

        /*!
        \brief Get access to a layer descriptor
        */
        const VisibleLayerDesc &getLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        /*!
        \brief Get the predictions
        */
        const DoubleBuffer2D &getHiddenStates() const {
            return _hiddenStates;
        }

        /*!
        \brief Get the hidden size
        */
        cl_int2 getHiddenSize() const {
            return _hiddenSize;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::PredictorLayer* fbPredictorLayer, ComputeSystem &cs);
        flatbuffers::Offset<schemas::PredictorLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };
}
