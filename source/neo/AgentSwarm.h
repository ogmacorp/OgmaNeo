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

            cl_int2 _chunkSize;
            cl_float _chunkGamma;
            //!@}

            /*!
            \brief Initialize defaults
            */
            AgentLayerDesc()
                : _radius(12), _qAlpha(0.1f),
                _qGamma(0.99f), _qLambda(0.98f),
                _epsilon(0.08f),
                _chunkSize({ 8, 8 }),
                _chunkGamma(0.5f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::AgentSwarmLayerDesc* fbAgentSwarmLayerDesc);
            flatbuffers::Offset<schemas::AgentSwarmLayerDesc> save(flatbuffers::FlatBufferBuilder &builder);
            //!@}
        };

    private:
        /*!
        \brief Feature hierarchy with same dimensions as swarm layer
        */
        Predictor _p;

        //!@{
        /*!
        \brief Layers and descs
        2D since layers can produce actions for multiple targets
        */
        std::vector<std::vector<AgentLayer>> _aLayers;
        std::vector<std::vector<AgentLayerDesc>> _aLayerDescs;
        std::vector<float> _rewardSums;
        std::vector<float> _rewardCounts;
        //!@}

        /*!
        \brief All ones image for first layer modulation
        */
        std::vector<cl::Image2D> _ones;

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
        void createRandom(ComputeSystem &cs, ComputeProgram &hProgram, ComputeProgram &pProgram, ComputeProgram &asProgram,
            const std::vector<cl_int2> &actionSizes, const std::vector<cl_int2> actionTileSizes,
            const std::vector<std::vector<AgentLayerDesc>> &aLayerDescs,
            const std::vector<Predictor::PredLayerDesc> &pLayerDescs,
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
        void simStep(ComputeSystem &cs, float reward, const std::vector<cl::Image2D> &inputs, const std::vector<cl::Image2D> &inputsCorrupted, std::mt19937 &rng, bool learn = true);

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
        const std::vector<AgentLayer> &getAgentLayer(int index) const {
            return _aLayers[index];
        }

        /*!
        \brief Get access to an agent layer descriptor
        */
        const std::vector<AgentLayerDesc> &getAgentLayerDesc(int index) const {
            return _aLayerDescs[index];
        }

        /*!
        \brief Get the actions
        Returns float 2D image where each element is actually an integer, representing the index of the select action for each tile.
        To get continuous values, divide each tile index by the number of elements in a tile (actionTileSize.x * actionTileSize.y).
        */
        const cl::Image2D &getAction(int index) const {
            return _aLayers.front()[index].getActions()[_back];
        }

        /*!
        \brief Get the underlying feature hierarchy
        */
        Predictor &getPredictor() {
            return _p;
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