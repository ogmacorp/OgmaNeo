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
#include "schemas/SparseFeaturesDistance_generated.h"

namespace ogmaneo {
    /*!
    \brief Distance encoder (sparse features)
    Learns a sparse code that is then used to predict the next input. Can be used with multiple layers
    */
    class OGMA_API SparseFeaturesDistance : public SparseFeatures {
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
            \brief Number of samples
            */
            cl_int _numSamples;

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
            \brief Short trace rate
            */
            float _lambda;

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 36, 36 }), _numSamples(3), _radius(8), _ignoreMiddle(false),
                _weightAlpha(1.0f), _lambda(0.8f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleDistanceLayerDesc* fbVisibleDistanceLayer, ComputeSystem &cs);
            schemas::VisibleDistanceLayerDesc save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Visible layer
        */
        struct VisibleLayer {
            /*!
            \brief Derived inputs
            */
            DoubleBuffer2D _derivedInputs;

            /*!
            \brief Samples (time sliced derived inputs)
            */
            DoubleBuffer3D _samples;

            /*!
            \brief Sample accumulation buffer
            */
            DoubleBuffer3D _samplesAccum;

            /*!
            \brief 2D buffer for retrieve a slice of samples
            */
            cl::Image2D _samplesSlice;

            /*!
            \brief Weights
            */
            DoubleBuffer3D _weights;

            //!@{
            /*!
            \brief Transformations
            */
            cl_float2 _hiddenToVisible;
            cl_float2 _visibleToHidden;

            cl_float2 _chunkToVisible;

            cl_int2 _reverseRadii;
            //!@}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleDistanceLayer* fbVisibleDistanceLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::VisibleDistanceLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Sparse Features Distance Descriptor
        */
        class OGMA_API SparseFeaturesDistanceDesc : public SparseFeatures::SparseFeaturesDesc {
        public:
            //!@{
            /*!
            \brief Construction information
            */
            std::shared_ptr<ComputeSystem> _cs;
            std::shared_ptr<ComputeProgram> _sfdProgram;
            std::vector<VisibleLayerDesc> _visibleLayerDescs;
            cl_int2 _hiddenSize;
            cl_int2 _chunkSize;
            float _gamma;
            cl_float2 _initWeightRange;
            std::mt19937 _rng;
            //!@}

            /*!
            \brief Defaults
            */
            SparseFeaturesDistanceDesc()
                : _hiddenSize({ 36, 36 }),
                _chunkSize({ 6, 6 }),
                _gamma(0.9999f),
                _initWeightRange({ 0.0f, 0.01f }),
                _rng()
            {
                _name = "distance";
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
                return std::make_shared<SparseFeaturesDistance>(*_cs, *_sfdProgram, _visibleLayerDescs, _hiddenSize, _chunkSize, _gamma, _initWeightRange, _rng);
            }

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::SparseFeaturesDistanceDesc* fbSparseFeaturesDistanceDesc, ComputeSystem &cs);
            flatbuffers::Offset<schemas::SparseFeaturesDistanceDesc> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        //!@{
        /*!
        \brief Hidden states, activations, chunk winners
        */
        DoubleBuffer2D _hiddenStates;
        DoubleBuffer2D _hiddenActivations;
        DoubleBuffer2D _chunkWinners;
        //!@}

        /*!
        \brief Hidden size
        */
        cl_int2 _hiddenSize;

        /*!
        \brief Size of chunks
        */
        cl_int2 _chunkSize;

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
        cl::Kernel _activateKernel;
        cl::Kernel _inhibitKernel;
        cl::Kernel _inhibitOtherKernel;
        cl::Kernel _learnWeightsKernel;
        cl::Kernel _deriveInputsKernel;
        cl::Kernel _sumKernel;
        cl::Kernel _sliceKernel;
        //!@}

    public:
        //!@{
        /*!
        \brief Additional parameters
        */
        float _gamma;
        //!@}

        /*!
        \brief Default constructor
        */
        SparseFeaturesDistance() {};

        /*!
        \brief Create a comparison sparse coder with random initialization
        \param sfdProgram program containing the distance encoder kernels.
        \param visibleLayerDescs descriptors for each input layer.
        \param hiddenSize hidden layer (SDR) size (2D).
        \param chunkSize chunk layer (SDR) size (2D).
        \param gamma decay of inverse learning rates (should be in the (1.0, 1.0] range for most purposes).
        \param initWeightRange range to initialize the weights into.
        \param rng a random number generator.
        */
        SparseFeaturesDistance(ComputeSystem &cs, ComputeProgram &sfdProgram,
            const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            cl_int2 hiddenSize,
            cl_int2 chunkSize,
            float gamma,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Add a new sample
        */
        void subSample(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, std::mt19937 &rng) override;

        /*!
        \brief Retrieve sample
        */
        cl::Image2D &getSubSample(ComputeSystem &cs, int vli, int index, std::mt19937 &rng) override;

        /*!
        \brief Retrieve sample
        */
        cl::Image2D &getSubSampleAccum(ComputeSystem &cs, int vli, int index, std::mt19937 &rng) override;

        /*!
        \brief Activate predictor
        */
        void activate(ComputeSystem &cs, std::mt19937 &rng) override;

        /*!
        \brief End a simulation step
        */
        void stepEnd(ComputeSystem &cs) override;

        /*!
        \brief Learning
        */
        void learn(ComputeSystem &cs, std::mt19937 &rng) override;

        /*!
        \brief Inhibit activations
        Inhibits given activations using this encoder's inhibitory pattern
        */
        void inhibit(ComputeSystem &cs, const cl::Image2D &activations, cl::Image2D &states, std::mt19937 &rng);

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
        \brief Get hidden size
        */
        cl_int2 getChunkSize() const override {
            return _chunkSize;
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
        \brief Get hidden activations
        */
        const DoubleBuffer2D &getHiddenActivations() const {
            return _hiddenActivations;
        }

        /*!
        \brief Get hidden chunk winner
        */
        const DoubleBuffer2D &getDistanceWinners() const {
            return _chunkWinners;
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