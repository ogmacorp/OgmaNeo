// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "OgmaNeo.h"
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
            \brief Predictor layer properties.
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
                : _radius(8), _alpha(0.08f), _beta(0.16f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::predictor::LayerDesc* fbLayerDesc);
            schemas::predictor::LayerDesc save(flatbuffers::FlatBufferBuilder &builder);
            //!@}
        };

    private:
        /*!
        \brief Feature hierarchy
        */
        FeatureHierarchy _h;

        /*!
        \brief Size of input
        */
        cl_int2 _inputSize;

        /*!
        \brief Layer descs
        */
        std::vector<PredLayerDesc> _pLayerDescs;

        /*!
        \brief Layers
        */
        std::vector<PredictorLayer> _pLayers;

    public:
        /*!
        \brief Create a sparse predictive hierarchy with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param inputSize size of the (2D) input.
        \param pLayerDescs Predictor layer descriptors.
        \param hLayerDescs Feature hierarchy layer descriptors.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        \param firstLearningRateScalar since the first layer predicts without thresholding while all others predict with it,
        the learning rate is scaled by this parameter for that first layer. Set to 1 if you want your pre-set learning rate
        to remain unchanged.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &program,
            cl_int2 inputSize, const std::vector<PredLayerDesc> &pLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
            cl_float2 initWeightRange, std::mt19937 &rng, float firstLearningRateScalar = 0.1f);

        /*!
        \brief Simulation step of hierarchy
        \param cs is the ComputeSystem.
        \param input input to the hierarchy (2D).
        \param inputCorrupted in many cases you can pass in the same value as for input, but in some cases you can also pass
        a corrupted version of the input for a "denoising auto-encoder" style effect on the learned weights.
        \param rng a random number generator.
        \param learn optional argument to disable learning.
        */
        void simStep(ComputeSystem &cs, const cl::Image2D &input, const cl::Image2D &inputCorrupted, std::mt19937 &rng, bool learn = true);

        /*!
        \brief Get number of predictor layers
        Matches the number of layers in the feature hierarchy.
        */
        size_t getNumPLayers() const {
            return _pLayers.size();
        }

        /*!
        \brief Get access to a predictor layer
        */
        const PredictorLayer &getPLayer(int index) const {
            return _pLayers[index];
        }

        /*!
        \brief Get access to a predictor layer desc
        */
        const PredLayerDesc &getPLayerDesc(int index) const {
            return _pLayerDescs[index];
        }

        /*!
        \brief Get the predictions
        */
        const cl::Image2D &getPrediction() const {
            return _pLayers.front().getHiddenStates()[_back];
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
        void load(const schemas::predictor::Predictor* fbPredictor, ComputeSystem &cs);
        flatbuffers::Offset<schemas::predictor::Predictor> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        
        void load(ComputeSystem &cs, ComputeProgram& prog, const std::string &fileName);
        void save(ComputeSystem &cs, const std::string& fileName);
        //!@}
    };
}