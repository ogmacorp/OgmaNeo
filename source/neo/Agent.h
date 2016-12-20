// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "AgentSwarm.h"
#include "Architect.h"
#include "schemas/Agent_generated.h"

namespace ogmaneo {
    // Declarations required for SWIG
    class AgentSwarm;

    /*!
    \brief Default Agent implementation (AgentSwarm)
    */
    class OGMA_API Agent {
    private:
        /*!
        \brief Internal OgmaNeo agent
        */
        AgentSwarm _as;

        std::mt19937 _rng;

        std::vector<cl::Image2D> _inputImages;
        std::vector<cl::Image2D> _corruptedInputImages;

        std::vector<ValueField2D> _actions;

        std::shared_ptr<Resources> _resources;

        std::vector<std::shared_ptr<ComputeProgram>> _programs;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::Agent* fbAgent, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::Agent> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}

    public:
        /*!
        \brief Run a single simulation tick
        */
        void simStep(float reward, std::vector<ValueField2D> &inputs, bool learn = true);
        void simStep(float reward, std::vector<ValueField2D> &inputs, std::vector<ValueField2D> &corruptedInputs, bool learn = true);

        /*!
        \brief Get the action vector
        */
        const std::vector<ValueField2D> &getActions() const {
            return _actions;
        }

        /*!
        \brief Access underlying AgentSwarm
        */
        AgentSwarm &getAgentSwarm() {
            return _as;
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