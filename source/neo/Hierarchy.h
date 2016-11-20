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

        std::vector<cl::Image2D> _inputImages;

        std::vector<ValueField2D> _predictions;

        std::shared_ptr<Resources> _resources;

        std::vector<PredictorLayer> _readoutLayers;

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
        void simStep(std::vector<ValueField2D> &inputs, bool learn = true);

        /*!
        \brief Get the action vector
        */
        const std::vector<ValueField2D> &getPredictions() const {
            return _predictions;
        }

        /*!
        \brief Access underlying Predictor
        */
        Predictor &getPredictor() {
            return _p;
        }

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