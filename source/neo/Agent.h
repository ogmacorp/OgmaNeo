// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "AgentSwarm.h"
#include "LayerDescs.h"
#include "schemas/Agent_generated.h"

namespace ogmaneo {
    /*!
    \brief Default Agent implementation (AgentSwarm)
    */
    class Agent {
    private:
        /*!
        \brief Internal OgmaNeo agent
        */
        AgentSwarm _as;

        int _inputWidth, _inputHeight;
        int _actionWidth, _actionHeight;
        int _actionTileWidth, _actionTileHeight;

        std::mt19937 _rng;

        cl::Image2D _inputImage;

        std::vector<float> _action; // Exploratory action (normalized float [0, 1])

        ComputeSystem* _pCs;

        void load(const schemas::agent::Agent* fbA, ComputeSystem &cs);
        flatbuffers::Offset<schemas::agent::Agent> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);

    public:
        /*!
        \brief Create the Agent
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param inputWidth is the (2D) width of the input layer.
        \param inputHeight is the (2D) height of the input layer.
        \param actionWidth is the (2D) width of the action layer.
        \param actionHeight is the (2D) height of the action layer.
        \param actionTileWidth is the (2D) width of each action tile (square one-hot action region).
        \param actionTileHeight is the (2D) height of each action tile (square one-hot action region).
        \param actionRadius is the radius onto the input action layer.
        \param layerDescs provide layer descriptors for hierachy and agent.
        \param initMinWeight is the minimum value for weight initialization.
        \param initMaxWeight is the maximum value for weight initialization.
        \param seed a random number generator seed.
        */
        Agent(ComputeSystem &cs, ComputeProgram &program,
            int inputWidth, int inputHeight,
            int actionWidth, int actionHeight,
            int actionTileWidth, int actionTileHeight,
            int actionRadius,
            const std::vector<LayerDescs> &layerDescs,
            float initMinWeight, float initMaxWeight, int seed);

        /*!
        \brief Run a single simulation tick
        */
        void simStep(float reward, const std::vector<float> &inputs, bool learn);

        /*!
        \brief Get the action vector
        */
        const std::vector<float> &getAction() const {
            return _action;
        }

        /*!
        \brief Get the hidden states for a layer
        \param[in] li Layer index.
        */
        std::vector<float> getStates(int li);

        //!@{
        /*!
        \brief Serialization
        */
        void load(ComputeSystem &cs, ComputeProgram &prog, const std::string &fileName);
        void save(ComputeSystem &cs, const std::string &fileName);
        //!@}
    };
}