// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "SparseFeatures.h"
#include "schemas/SparseFeaturesDelay_generated.h"

namespace ogmaneo {
    /*!
    \brief Delay encoder (sparse features)
    Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
    */
    class OGMA_API SparseFeaturesDelay : public SparseFeatures {
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
            \brief Fast trace decay
            */
            cl_float _lambda;

            /*!
            \brief Slow trace decay
            */
            cl_float _gamma;

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 8, 8 }), _radius(6), _ignoreMiddle(false),
                _weightAlpha(0.0f), _lambda(0.9f), _gamma(0.96f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleDelayLayerDesc* fbVisibleDelayLayerDesc, ComputeSystem &cs);
            schemas::VisibleDelayLayerDesc save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
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

            /*!
            \brief Weights
            */
            DoubleBuffer3D _weights; // Encoding weights (creates spatio-temporal sparse code)

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
            void load(const schemas::VisibleDelayLayer* fbVisibleDelayLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::VisibleDelayLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Sparse Features Chunk Descriptor
        */
        class OGMA_API SparseFeaturesDelayDesc : public SparseFeatures::SparseFeaturesDesc {
        public:
            //!@{
            /*!
            \brief Construction information
            */
            std::shared_ptr<ComputeSystem> _cs;
            std::shared_ptr<ComputeProgram> _sfcProgram;
            std::vector<VisibleLayerDesc> _visibleLayerDescs;
            cl_int2 _hiddenSize;
            cl_int _inhibitionRadius;
            cl_float _biasAlpha;
            cl_float _activeRatio;
            cl_float2 _initWeightRange;
            std::mt19937 _rng;
            //!@}

            /*!
            \brief Defaults
            */
            SparseFeaturesDelayDesc()
                : _hiddenSize({ 16, 16 }),
                _inhibitionRadius(6),
                _biasAlpha(0.01f), _activeRatio(0.01f),
                _initWeightRange({ -0.01f, 0.01f }),
                _rng()
            {
                _name = "delay";
            }

            size_t getNumVisibleLayers() const override {
                return _visibleLayerDescs.size();
            }

            cl_int2 getVisibleLayerSize(int vli) const override {
                return _visibleLayerDescs[vli]._size;
            }

            cl_int2 getHiddenSize() const override {
                return _hiddenSize;
            }

            /*!
            \brief Factory
            */
            std::shared_ptr<SparseFeatures> sparseFeaturesFactory() override {
                return std::make_shared<SparseFeaturesDelay>(*_cs, *_sfcProgram, _visibleLayerDescs, _hiddenSize, _inhibitionRadius, _biasAlpha, _activeRatio, _initWeightRange, _rng);
            }

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::SparseFeaturesDelayDesc* fbSparseFeaturesDelayDesc, ComputeSystem &cs);
            flatbuffers::Offset<schemas::SparseFeaturesDelayDesc> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        //!@{
        /*!
        \brief Activations, states, biases
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
        \brief Inhibition radius
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
        //!@{
        /*!
        \brief Additional parameters
        */
        cl_float _biasAlpha;
        cl_float _activeRatio;
        //!@}

        /*!
        \brief Default constructor
        */
        SparseFeaturesDelay() {};

        /*!
        \brief Create a comparison sparse coder with random initialization
        Requires the compute system, program with the NeoRL kernels, and initialization information.
        \param visibleLayerDescs descriptors for each input layer.
        \param hiddenSize hidden layer (SDR) size (2D).
        \param rng a random number generator.
        */
        SparseFeaturesDelay(ComputeSystem &cs, ComputeProgram &sfdProgram,
            const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            cl_int2 hiddenSize, cl_int inhibitionRadius,
            cl_float biasAlpha,
            cl_float activeRatio,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Activate predictor
        \param lambda decay of hidden unit traces.
        \param activeRatio % active units.
        \param rng a random number generator.
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &predictionsPrev, std::mt19937 &rng) override;

        /*!
        \brief End a simulation step
        */
        void stepEnd(ComputeSystem &cs) override;

        /*!
        \brief Learning
        \param biasAlpha learning rate of bias.
        \param activeRatio % active units.
        \param gamma synaptic trace decay.
        */
        void learn(ComputeSystem &cs, const cl::Image2D &predictionsPrev, std::mt19937 &rng) override;

        /*!
        \brief Inhibit
        Inhibits given activations using this encoder's inhibitory pattern
        */
        void inhibit(ComputeSystem &cs, const cl::Image2D &activations, cl::Image2D &states, std::mt19937 &rng) override;

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
        cl_int2 getHiddenSize() const override {
            return _hiddenSize;
        }

        /*!
        \brief Get hidden states
        */
        const DoubleBuffer2D &getHiddenStates() const override {
            return _hiddenStates;
        }

        /*!
        \brief Get hidden activations
        */
        const DoubleBuffer2D &getHiddenActivations() const {
            return _hiddenActivations;
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
        void clearMemory(ComputeSystem &cs) override;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::SparseFeatures* fbSparseFeatures, ComputeSystem &cs) override;
        flatbuffers::Offset<schemas::SparseFeatures> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) override;
        //!@}
    };
}