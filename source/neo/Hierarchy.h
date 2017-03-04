// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "Predictor.h"
#include "Architect.h"
#include "schemas/Hierarchy_generated.h"

namespace ogmaneo {
    // Declarations required for SWIG
    class ValueField2D;
    class Predictor;
    class PredictorLayer;

    /*!
    \brief Default Hierarchy implementation (Predictor)
    */
    class OGMA_API Hierarchy {
    private:
        /*!
        \brief Internal OgmaNeo agent
        */
        Predictor _p;

        std::mt19937 _rng;

        std::vector<cl::Image2D> _inputImagesFeed;
        std::vector<cl::Image2D> _inputImagesPredict;

        std::vector<ValueField2D> _predictions;

        std::shared_ptr<Resources> _resources;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::Hierarchy* fbHierarchy, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::Hierarchy> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}

    public:
        /*!
        \brief Run a single simulation tick
        */
        void activate(std::vector<ValueField2D> &inputsFeed);
        void learn(std::vector<ValueField2D> &inputsPredict, float tdError = 0.0f);

        /*!
        \brief Get the feed images (input)
        */
        const std::vector<cl::Image2D> &getInputImagesFeed() const {
            return _inputImagesFeed;
        }

        /*!
        \brief Get the prediction target images
        */
        const std::vector<cl::Image2D> &getInputImagesPredict() const {
            return _inputImagesPredict;
        }

        /*!
        \brief Get the predictions
        */
        std::vector<ValueField2D> &getPredictions() {
            return _predictions;
        }

        /*!
        \brief Access underlying Predictor
        */
        Predictor &getPredictor() {
            return _p;
        }

        /*!
        \brief Specifically for accessing chunk states from bindings
        */
        void readChunkStates(int li, ValueField2D &valueField);

        //!@{
        /*!
        \brief Serialization
        */
        void load(ComputeSystem &cs, const std::string &fileName);
        void save(ComputeSystem &cs, const std::string &fileName);
        //!@}

        friend class Architect;
    };
}