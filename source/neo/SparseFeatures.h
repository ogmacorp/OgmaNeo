// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "OgmaNeo.h"
#include "schemas/SparseFeatures_generated.h"

namespace ogmaneo {
    /*!
    \brief Sparse predictor
    Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
    */
    class OGMA_API SparseFeatures {
    public:
        /*!
        \brief Visible layer desc
        */
        struct VisibleLayerDesc {
            /*!
            \brief Size of layer
            */
            cl_int2 _size;

            /*!
            \brief Radius onto input
            */
            cl_int _radius;

            /*!
            \brief Whether or not the middle (center) input should be ignored (self in recurrent schemes)
            */
            unsigned char _ignoreMiddle;

            /*!
            \brief Learning rate
            */
            cl_float _weightAlpha;

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 8, 8 }), _radius(6), _ignoreMiddle(false),
                _weightAlpha(0.004f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::hierarchy::VisibleLayerDesc* fbVisibleLayerDesc, ComputeSystem &cs);
            schemas::hierarchy::VisibleLayerDesc save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Visible layer
        */
        struct VisibleLayer {
            /*!
            \brief Possibly manipulated input
            */
            DoubleBuffer2D _derivedInput;

            //!@{
            /*!
            \brief Weights
            */
            DoubleBuffer3D _weights; // Encoding weights (creates spatio-temporal sparse code)
            //!@}

            //!@{
            /*!
            \brief Transformations
            */
            cl_float2 _hiddenToVisible;
            cl_float2 _visibleToHidden;

            cl_int2 _reverseRadii;
            //!@}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::hierarchy::VisibleLayer* fbVisibleLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::hierarchy::VisibleLayer> save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs);
            //!@}
        };

    private:
        //!@{
        /*!
        \brief Hidden activations, states, biases, errors, predictions
        */
        DoubleBuffer2D _hiddenActivations;
        DoubleBuffer2D _hiddenStates;
        DoubleBuffer2D _hiddenBiases;
        //!@}

        /*!
        \brief Hidden size
        */
        cl_int2 _hiddenSize;

        /*!
        \brief Lateral inhibitory radius
        */
        cl_int _inhibitionRadius;

        /*!
        \brief Hidden summation temporary buffer
        */
        DoubleBuffer2D _hiddenSummationTemp;

        //!@{
        /*!
        \brief Layers and descs
        */
        std::vector<VisibleLayerDesc> _visibleLayerDescs;
        std::vector<VisibleLayer> _visibleLayers;
        //!@}

        //!@{
        /*!
        \brief Kernels
        */
        cl::Kernel _stimulusKernel;
        cl::Kernel _activateKernel;
        cl::Kernel _inhibitKernel;
        cl::Kernel _learnWeightsKernel;
        cl::Kernel _learnBiasesKernel;
        cl::Kernel _deriveInputsKernel;
        //!@}

    public:
        /*!
        \brief Create a comparison sparse coder with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param visibleLayerDescs descriptors for each input layer.
        \param hiddenSize hidden layer (SDR) size (2D).
        \param inhibitionRadius inhibitory radius.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &program,
            const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            cl_int2 hiddenSize,
            cl_int inhibitionRadius,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Activate predictor
        \param cs is the ComputeSystem.
        \param visibleStates the input layer states.
        \param lambda decay of hidden unit traces.
        \param activeRatio % active units.
        \param rng a random number generator.
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio, std::mt19937 &rng);

        /*!
        \brief End a simulation step
        */
        void stepEnd(ComputeSystem &cs);

        /*!
        \brief Learning
        \param cs is the ComputeSystem.
        \param biasAlpha learning rate of bias.
        \param activeRatio % active units.
        \param gamma synaptic trace decay.
        */
        void learn(ComputeSystem &cs, float biasAlpha, float activeRatio);

        /*!
        \brief Get number of visible layers
        */
        size_t getNumVisibleLayers() const {
            return _visibleLayers.size();
        }

        /*!
        \brief Get access to visible layer
        */
        const VisibleLayer &getVisibleLayer(int index) const {
            return _visibleLayers[index];
        }

        /*!
        \brief Get access to visible layer
        */
        const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        /*!
        \brief Get hidden size
        */
        cl_int2 getHiddenSize() const {
            return _hiddenSize;
        }

        /*!
        \brief Get hidden states
        */
        const DoubleBuffer2D &getHiddenStates() const {
            return _hiddenStates;
        }

        /*!
        \brief Get hidden biases
        */
        const DoubleBuffer2D &getHiddenBiases() const {
            return _hiddenBiases;
        }

        /*!
        \brief Clear the working memory
        */
        void clearMemory(ComputeSystem &cs);

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::hierarchy::SparseFeatures* fbSF, ComputeSystem &cs);
        flatbuffers::Offset<schemas::hierarchy::SparseFeatures> save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs);
        //!@}
    };

}