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
#include "schemas/SparseFeaturesReLU_generated.h"

namespace ogmaneo {
    /*!
    \brief ReLU encoder (sparse features)
    Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
    */
    class OGMA_API SparseFeaturesReLU : public SparseFeatures {
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
            cl_int _radiusHidden;

            /*!
            \brief Radius onto hidden
            */
            cl_int _radiusVisible;

            /*!
            \brief Whether or not the middle (center) input should be ignored (self in recurrent schemes)
            */
            unsigned char _ignoreMiddle;

            //!@{
            /*!
            \brief Learning rates
            */
            cl_float _weightAlphaHidden;
            cl_float _weightAlphaVisible;
            //!@}

            /*!
            \brief Short trace rate
            */
            float _lambda;

            /*!
            \brief Whether this layer should be predicted
            */
            bool _predict;

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 8, 8 }), _radiusHidden(8), _radiusVisible(8), _ignoreMiddle(false),
                _weightAlphaHidden(0.005f), _weightAlphaVisible(0.05f), _lambda(0.8f), _predict(true)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleReLULayerDesc* fbVisibleReLULayer, ComputeSystem &cs);
            schemas::VisibleReLULayerDesc save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
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
            \brief Predictions
            */
            DoubleBuffer2D _predictions;

            /*!
            \brief Samples (time sliced derived inputs)
            */
            DoubleBuffer3D _samples;

            //!@{
            /*!
            \brief Weights
            */
            DoubleBuffer3D _weightsHidden;
            DoubleBuffer3D _weightsVisible;
            //!@}

            //!@{
            /*!
            \brief Transformations
            */
            cl_float2 _hiddenToVisible;
            cl_float2 _visibleToHidden;

            cl_int2 _reverseRadiiHidden;
            cl_int2 _reverseRadiiVisible;
            //!@}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleReLULayer* fbVisibleReLULayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::VisibleReLULayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Sparse Features ReLU Descriptor
        */
        class OGMA_API SparseFeaturesReLUDesc : public SparseFeatures::SparseFeaturesDesc {
        public:
            //!@{
            /*!
            \brief Construction information
            */
            std::shared_ptr<ComputeSystem> _cs;
            std::shared_ptr<ComputeProgram> _sfrProgram;
            std::vector<VisibleLayerDesc> _visibleLayerDescs;
            cl_int2 _hiddenSize;
            int _numSamples;
            cl_int _lateralRadius;
            cl_float _gamma;
            cl_float _activeRatio;
            cl_float _biasAlpha;
            cl_float2 _initWeightRange;
            std::mt19937 _rng;
            //!@}

            /*!
            \brief Defaults
            */
            SparseFeaturesReLUDesc()
                : _hiddenSize({ 16, 16 }),
                _numSamples(1), _lateralRadius(6),
                _gamma(0.92f), _activeRatio(0.02f), _biasAlpha(0.005f),
                _initWeightRange({ -0.01f, 0.01f }),
                _rng()
            {
                _name = "ReLU";
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
                return std::make_shared<SparseFeaturesReLU>(*_cs, *_sfrProgram, _visibleLayerDescs, _hiddenSize, _numSamples, _lateralRadius, _gamma, _activeRatio, _biasAlpha, _initWeightRange, _rng);
            }

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::SparseFeaturesReLUDesc* fbSparseFeaturesReLUDesc, ComputeSystem &cs);
            flatbuffers::Offset<schemas::SparseFeaturesReLUDesc> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        //!@{
        /*!
        \brief Hidden states, biases
        */
        DoubleBuffer2D _hiddenStates;
        DoubleBuffer2D _hiddenBiases;
        //!@}

        /*!
        \brief Hidden size
        */
        cl_int2 _hiddenSize;

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
        cl::Kernel _addSampleKernel;
        cl::Kernel _stimulusKernel;
        cl::Kernel _inhibitKernel;
        cl::Kernel _predictKernel;
        cl::Kernel _inhibitOtherKernel;
        cl::Kernel _errorPropKernel;
        cl::Kernel _learnWeightsHiddenKernel;
        cl::Kernel _learnWeightsVisibleKernel;
        cl::Kernel _learnBiasesKernel;
        cl::Kernel _deriveInputsKernel;
        //!@}

    public:
        //!@{
        /*!
        \brief Additional parameters
        */
        int _numSamples;
        int _lateralRadius;
        float _gamma;
        float _activeRatio;
        float _biasAlpha;
        //!@}

        /*!
        \brief Default constructor
        */
        SparseFeaturesReLU() {};

        /*!
        \brief Create a comparison sparse coder with random initialization
        Requires the compute system, program with the NeoRL kernels, and initialization information.
        \param visibleLayerDescs descriptors for each input layer.
        \param hiddenSize hidden layer (SDR) size (2D).
        \param rng a random number generator.
        */
        SparseFeaturesReLU(ComputeSystem &cs, ComputeProgram &sfrProgram,
            const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
            int numSamples, int lateralRadius,
            cl_float gamma, cl_float activeRatio, cl_float biasAlpha,
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
        \brief Get hidden states
        */
        const cl::Image2D &getHiddenContext() const override {
            return _hiddenStates[_back];
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