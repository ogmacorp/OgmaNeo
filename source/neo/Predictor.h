// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "FeatureHierarchy.h"
#include "PredictorLayer.h"
#include "schemas/Predictor_generated.h"

namespace ogmaneo {
    /*!
    \brief Predicts temporal streams of data
    Combines the bottom-up feature hierarchy with top-down predictions.
    */
    class OGMA_API Predictor {
    public:
        /*!
        \brief Description of a predictor layer
        */
        struct PredLayerDesc {
            //!@{
            /*!
            \brief Predictor layer properties
            Is Q layer, radius onto hidden layer, learning rates for feed-forward and feed-back, trace decays.
            */
            bool _isQ;

            int _radius;
            float _alpha;
            float _beta;
            float _lambda;
            float _gamma;
            //!@}

            /*!
            \brief Initialize defaults
            */
            PredLayerDesc()
                : _isQ(false), _radius(8), _alpha(0.02f), _beta(0.04f), _lambda(0.98f), _gamma(0.99f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::PredLayerDesc* fbPredLayerDesc, ComputeSystem &cs);
            flatbuffers::Offset<schemas::PredLayerDesc> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        /*!
        \brief Feature hierarchy
        */
        FeatureHierarchy _h;

        /*!
        \brief Layer descs
        */
        std::vector<std::vector<PredLayerDesc>> _pLayerDescs; // 2D since each layer can predict multiple inputs

        /*!
        \brief Layers
        */
        std::vector<std::vector<PredictorLayer>> _pLayers; // 2D since each layer can predict multiple inputs

        /*!
        \brief Whether reset last tick
        */
        std::vector<bool> _needsUpdate;

    public:
        /*!
        \brief Create a sparse predictive hierarchy with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param hProgram is the ComputeProgram associated with the ComputeSystem and loaded with the hierarchy kernel code.
        \param pProgram is the ComputeProgram associated with the ComputeSystem and loaded with the predictor kernel code.
        \param inputSizes Size of each input layer.
        \param inputChunkSizes Size of input chunks if applicable, 0 otherwise.
        \param pLayerDescs Predictor layer descriptors.
        \param hLayerDescs Feature hierarchy layer descriptors.
        \param initWeightRange range to initialize predictors.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &hProgram, ComputeProgram &pProgram,
            const std::vector<cl_int2> &inputSizes, const std::vector<cl_int2> &inputChunkSizes,
            const std::vector<std::vector<PredLayerDesc>> &pLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Activation step of hierarchy
        \param cs is the ComputeSystem.
        \param inputsFeed input to the hierarchy (2D).
        \param rng a random number generator.
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &inputsFeed, std::mt19937 &rng);

        /*!
        \brief Learning step of hierarchy
        \param cs is the ComputeSystem.
        \param inputsPredict input to the hierarchy (2D) that will be used as a prediction target.
        \param rng a random number generator.
        \param tdError optional TD error argument for reinforcement learning.
        */
        void learn(ComputeSystem &cs, const std::vector<cl::Image2D> &inputsPredict, std::mt19937 &rng, float tdError = 0.0f);

        /*!
        \brief Get number of predictor layers
        Matches the number of layers in the feature hierarchy.
        */
        size_t getNumPredLayers() const {
            return _pLayers.size();
        }

        /*!
        \brief Get access to a predictor layer
        */
        const std::vector<PredictorLayer> &getPredLayer(int index) const {
            return _pLayers[index];
        }

        /*!
        \brief Get access to a predictor layer desc
        */
        const std::vector<PredLayerDesc> &getPredLayerDesc(int index) const {
            return _pLayerDescs[index];
        }

        /*!
        \brief Get the predictions
        */
        const DoubleBuffer2D &getPredictions(int index) const {
            return _pLayers.front()[index].getHiddenStates();
        }

        /*!
        \brief Get the underlying feature hierarchy
        */
        FeatureHierarchy &getHierarchy() {
            return _h;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::Predictor* fbPredictor, ComputeSystem &cs);
        flatbuffers::Offset<schemas::Predictor> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };
}