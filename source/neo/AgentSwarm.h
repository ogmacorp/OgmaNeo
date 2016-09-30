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
#include "AgentLayer.h"
#include "schemas/AgentSwarm_generated.h"

namespace ogmaneo {
    /*!
    \brief Swarm of agents routed through a feature hierarchy
    */
    class OGMA_API AgentSwarm {
    public:
        /*!
        \brief Layer desc for swarm layers
        */
        struct AgentLayerDesc {
            /*!
            \brief Radius of connections onto previous swarm layer
            */
            cl_int _radius;

            //!@{
            /*!
            \brief Q learning parameters
            */
            cl_float _qAlpha;
            cl_float _qGamma;
            cl_float _qLambda;
            cl_float _epsilon;
            //!@}

            /*!
            \brief Initialize defaults
            */
            AgentLayerDesc()
                : _radius(12), _qAlpha(0.00004f),
                _qGamma(0.99f), _qLambda(0.98f), _epsilon(0.06f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::AgentLayerDesc* fbAgentLayerDesc);
            schemas::AgentLayerDesc save(flatbuffers::FlatBufferBuilder &builder);
            //!@}
        };

    private:
        /*!
        \brief Feature hierarchy with same dimensions as swarm layer
        */
        FeatureHierarchy _h;

        //!@{
        /*!
        \brief Layers and descs
        */
        std::vector<AgentLayer> _aLayers;
        std::vector<AgentLayerDesc> _aLayerDescs;
        //!@}

        /*!
        \brief All ones image for first layer modulation
        */
        cl::Image2D _ones;

    public:
        /*!
        \brief Initialize defaults
        */
        AgentSwarm()
        {}

        /*!
        \brief Create a predictive hierarchy with random initialization. 
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param inputSize is the (2D) size of the input layer.
        \param actionSize is the (2D) size of the action layer.
        \param actionTileSize is the (2D) size of each action tile (square one-hot action region).
        \param actionRadius is the radius onto the input action layer.
        \param aLayerDescs are Agent layer descriptors.
        \param hLayerDescs are Feature hierarchy layer descriptors.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &program,
            cl_int2 inputSize, cl_int2 actionSize, cl_int2 actionTileSize, cl_int actionRadius,
            const std::vector<AgentLayerDesc> &aLayerDescs,
            const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
            cl_float2 initWeightRange, std::mt19937 &rng);

        /*!
        \brief Simulation step of hierarchy
        Takes reward and inputs, optionally disable learning.
        \param cs is the ComputeSystem.
        \param reward the reinforcement learning signal.
        \param input the input layer state.
        \param rng a random number generator.
        \param learn optional argument to disable learning.
        */
        void simStep(ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn = true);

        /*!
        \brief Get number of agent (swarm) layers
        Matches the number of feature hierarchy layers
        */
        size_t getNumAgentLayers() const {
            return _aLayers.size();
        }

        /*!
        \brief Get access to an agent layer
        */
        const AgentLayer &getAgentLayer(int index) const {
            return _aLayers[index];
        }

        /*!
        \brief Get access to an agent layer descriptor
        */
        const AgentLayerDesc &getAgentLayerDesc(int index) const {
            return _aLayerDescs[index];
        }

        /*!
        \brief Get the actions
        Returns float 2D image where each element is actually an integer, representing the index of the select action for each tile.
        To get continuous values, divide each tile index by the number of elements in a tile (actionTileSize.x * actionTileSize.y).
        */
        const cl::Image2D &getAction() const {
            return _aLayers.back().getActions()[_back];
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
        void load(const schemas::AgentSwarm* fbAgentSwarm, ComputeSystem &cs);
        flatbuffers::Offset<schemas::AgentSwarm> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };
}