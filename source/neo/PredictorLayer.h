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
        \brief Types of thresholding
        */
        enum Type {
            _none, _inhibitBinary, _q
        };

        /*!
        \brief Layer descriptor
        */
        struct VisibleLayerDesc {
            //!@{
            /*!
            \brief Layer properties
            Input size, radius onto input, learning rate, TD lambda, additional decay gamma.
            */
            cl_int2 _size;

            cl_int _radius;

            cl_float _alpha;
    
            cl_float _lambda; // For Q learning

            cl_float _gamma;
            //!@}

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 16, 16 }),
                _radius(8),
                _alpha(0.01f), _lambda(0.98f),
                _gamma(0.99f)
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
        \brief Type of sparsity/thresholding
        */
        Type _type;

        /*!
        \brief Size of the prediction
        */
        cl_int2 _hiddenSize;

        /*!
        \brief Size of chunks
        */
        cl_int2 _chunkSize;

        /*!
        \brief Hidden stimulus summation temporary buffer
        */
        DoubleBuffer2D _hiddenSummationTemp;

        //!@{
        /*!
        \brief Predictions
        */
        DoubleBuffer2D _hiddenStates;
        //!@}

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
        cl::Kernel _propagateKernel;
        cl::Kernel _inhibitBinaryKernel;
        //!@}

    public:
        /*!
        \brief Create a predictor layer with random initialization.
        \param cs is the ComputeSystem.
        \param plProgram is the ComputeProgram associated with the ComputeSystem and loaded with the predictor kernel code.
        \param hiddenSize size of the predictions (output).
        \param visibleLayerDescs are descriptors for visible layers.
        \param type inhibition/thresholding type.
        \param chunkSize size of a chunk, if applicable (else 0).
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &plProgram,
            cl_int2 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            Type type, cl_int2 chunkSize,
            cl_float2 initWeightRange, std::mt19937 &rng);

        /*!
        \brief Activate predictor (predict values)
        \param cs is the ComputeSystem.
        \param visibleStates the input layer states.
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, std::mt19937 &rng);

        /*!
        \brief Propagate (reconstruct)
        */
        void propagate(ComputeSystem &cs, const cl::Image2D &hiddenStates, const cl::Image2D &hiddenTargets, int vli, DoubleBuffer2D &visibleStates, std::mt19937 &rng);

        /*!
        \brief Learn predictor
        \param cs is the ComputeSystem.
        \param targets target values to update towards.
        \param predictFromPrevious whether to map from (t-1) to (t) or (t) to (t).
        \param tdError optional TD error for reinforcement learning.
        */
        void learn(ComputeSystem &cs, const cl::Image2D &targets, bool predictFromPrevious = true, float tdError = 0.0f);

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

        /*!
        \brief Get the hidden summation buffer
        */
        const DoubleBuffer2D &getHiddenSummation() const {
            return _hiddenSummationTemp;
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
