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
    \brief Predicts temporal streams of data.
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
            Radius onto hidden layer, learning rates for feed-forward and feed-back.
            */
            int _radius;
            float _alpha;
            float _beta;
            //!@}

            /*!
            \brief Initialize defaults
            */
            PredLayerDesc()
                : _radius(8), _alpha(0.1f), _beta(0.3f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::PredLayerDesc* fbPredLayerDesc, ComputeSystem &cs);
            schemas::PredLayerDesc save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
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
        std::vector<PredLayerDesc> _pLayerDescs;

        /*!
        \brief Layers
        */
        std::vector<PredictorLayer> _pLayers; // 2D since each layer can predict multiple inputs

    public:
        /*!
        \brief Create a sparse predictive hierarchy with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param pProgram is the ComputeProgram associated with the ComputeSystem and loaded with the predictor kernel code.
        \brief shouldPredictInput describes which of the bottom (input) layers should be predicted (have an associated predictor layer).
        \param pLayerDescs Predictor layer descriptors.
        \param hLayerDescs Feature hierarchy layer descriptors.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &hProgram, ComputeProgram &pProgram,
            const std::vector<PredLayerDesc> &pLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Simulation step of hierarchy
        \param cs is the ComputeSystem.
        \param input input to the hierarchy (2D).
        \param inputCorrupted in many cases you can pass in the same value as for input, but in some cases you can also pass
        a corrupted version of the input for a "denoising auto-encoder" style effect on the learned weights.
        \param rng a random number generator.
        \param learn optional argument to disable learning.
        */
        void simStep(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, const std::vector<cl::Image2D> &inputsCorrupted, std::mt19937 &rng, bool learn = true);

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
        const PredictorLayer &getPredLayer(int index) const {
            return _pLayers[index];
        }

        /*!
        \brief Get access to a predictor layer desc
        */
        const PredLayerDesc &getPredLayerDesc(int index) const {
            return _pLayerDescs[index];
        }

        /*!
        \brief Get the predictions
        */
        const DoubleBuffer2D &getHiddenPrediction() const {
            return _pLayers.front().getHiddenStates();
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