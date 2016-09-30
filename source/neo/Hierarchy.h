// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Predictor.h"
#include "LayerDescs.h"
#include "schemas/Hierarchy_generated.h"

namespace ogmaneo {
    /*!
    \brief Default Hierarchy implementation (FeatureHierarchy)
    */
    class Hierarchy {
    private:
        /*!
        \brief Internal OgmaNeo hiearchy
        */
        Predictor _ph;

        int _inputWidth, _inputHeight;

        std::mt19937 _rng;

        cl::Image2D _inputImage;

        /*!
        \brief Prediction vector
        */
        std::vector<float> _pred;

        ComputeSystem* _pCs;

        void load(const schemas::hierarchy::Hierarchy* fbFH, ComputeSystem &cs);
        flatbuffers::Offset<schemas::hierarchy::Hierarchy> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);

    public:
        /*!
        \brief Create the Hierarchy
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param inputWidth is the width of input to the hierarchy.
        \param inputHeight is the height of input to the hierarchy.
        \param layerDescs provide layer descriptors for hierachy.
        \param initMinWeight is the minimum value for weight initialization.
        \param initMaxWeight is the maximum value for weight initialization.
        \param seed a random number generator seed.
        */
        Hierarchy(ComputeSystem &cs, ComputeProgram &program,
            int inputWidth, int inputHeight,
            const std::vector<LayerDescs> &layerDescs,
            float initMinWeight, float initMaxWeight, int seed);

        /*!
        \brief Run a single simulation tick
        \param inputs the inputs to the bottom-most layer.
        \param learn optional argument to disable learning.
        */
        void simStep(const std::vector<float> &inputs, bool learn);

        /*!
        \brief Get the current prediction vector
        */
        const std::vector<float> &getPrediction() const {
            return _pred;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(ComputeSystem &cs, ComputeProgram &prog, const std::string &fileName);
        void save(ComputeSystem &cs, const std::string &fileName);
        //!@}
    };
}