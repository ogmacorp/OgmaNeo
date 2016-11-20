// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "system/ComputeSystem.h"
#include "system/ComputeProgram.h"
#include "Helpers.h"
#include "schemas/AgentLayer_generated.h"

namespace ogmaneo {
    /*!
    \brief Agent layer.
    Contains a (2D) swarm of small Q learning agents, one per action tile
    */
    class OGMA_API AgentLayer {
    public:
        /*!
        \brief Layer desc for inputs to the swarm layer
        */
        struct VisibleLayerDesc {
            //!@{
            /*!
            \brief Layer properties
            Size, radius onto layer, and learning rate
            */
            cl_int2 _size;

            cl_int _radius;

            cl_float _qAlpha;
            cl_float _actionAlpha;
            //!@}

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 16, 16 }),
                _radius(12),
                _qAlpha(0.001f),
                _actionAlpha(0.02f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleAgentLayerDesc* fbVisibleAgentLayerDesc);
            schemas::VisibleAgentLayerDesc save(flatbuffers::FlatBufferBuilder &builder);
            //!@}
        };

        /*!
        \brief Layer
        */
        struct VisibleLayer {
            //!@{
            /*!
            \brief Layer data
            */
            DoubleBuffer3D _qWeights;
            DoubleBuffer3D _actionWeights;

            cl_float2 _qToVisible;
            cl_float2 _visibleToQ;

            cl_int2 _reverseRadiiQ;

            cl_float2 _hiddenToVisible;
            cl_float2 _visibleToHidden;

            cl_int2 _reverseRadiiHidden;
            //!@}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::VisibleAgentLayer* fbVisibleAgentLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::VisibleAgentLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        /*!
        \brief Size of action layer in tiles
        */
        cl_int2 _numActionTiles;

        /*!
        \brief Size of an action tile
        */
        cl_int2 _actionTileSize;

        /*!
        \brief Size of the total action region (hidden size), (numActionTiles.x * actionTileSize.x, numActionTiles.y * actionTileSize.y)
        */
        cl_int2 _hiddenSize;

        //!@{
        /*!
        \brief Hidden state variables
        Q states, actions, td errors, one hot action
        */
        DoubleBuffer2D _qStates;
        DoubleBuffer2D _actionProbabilities;

        DoubleBuffer2D _actionTaken;
        cl::Image2D _tdError;
        cl::Image2D _oneHotAction;
        //!@}

        //!@{
        /*!
        \brief Hidden stimulus summation temporary buffers
        */
        DoubleBuffer2D _hiddenSummationTempQ;
        DoubleBuffer2D _hiddenSummationTempHidden;
        //!@}

        //!@{
        /*!
        \brief Layers and descs
        */
        std::vector<VisibleLayer> _visibleLayers;
        std::vector<VisibleLayerDesc> _visibleLayerDescs;
        //!@}

        //!@{
        /*!
        \brief Additional kernels
        */
        cl::Kernel _activateKernel;
        cl::Kernel _learnQKernel;
        cl::Kernel _learnActionsKernel;
        cl::Kernel _actionToOneHotKernel;
        cl::Kernel _getActionKernel;
        cl::Kernel _setActionKernel;
        //!@}

    public:
        /*!
        \brief Initialize defaults
        */
        AgentLayer()
        {}

        /*!
        \brief Create a predictive hierarchy with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param numActionTiles is the (2D) size of the action layer.
        \param actionTileSize is the (2D) size of each action tile (square one-hot action region).
        \param visibleLayerDescs is a vector of visible layer parameters.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &program,
            cl_int2 numActionTiles, cl_int2 actionTileSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Simulation step of agent layer agents.
        Requres several reinforcement learning parameters.
        \param cs is the ComputeSystem.
        \param reward the reinforcement learning signal.
        \param visibleStates all input layer states.
        \param modulator layer that modulates the agents in the swarm (1 = active, 0 = inactive).
        \param qGamma Q learning gamma.
        \param qLambda Q learning lambda (trace decay).
        \param epsilon Q learning epsilon greedy exploration rate.
        \param rng a random number generator.
        \param learn optional argument to disable learning.
        */
        void simStep(ComputeSystem &cs, float reward, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &modulator,
            float qGamma, float qLambda, float actionLambda, float maxActionWeightMag, std::mt19937 &rng, bool learn = true);

        /*!
        \brief Clear memory (recurrent data)
        \param cs is the ComputeSystem.
        */
        void clearMemory(ComputeSystem &cs);

        /*!
        \brief Get number of layers
        */
        size_t getNumLayers() const {
            return _visibleLayers.size();
        }

        /*!
        \brief Get access to a layer
        \param[in] index Visible layer index.
        */
        const VisibleLayer &getLayer(int index) const {
            return _visibleLayers[index];
        }

        /*!
        \brief Get access to a layer descriptor
        \param[in] index Visible layer descriptor index.
        */
        const VisibleLayerDesc &getLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        /*!
        \brief Get the Q states
        */
        const DoubleBuffer2D &getQStates() const {
            return _qStates;
        }

        /*!
        \brief Get the hidden action activations
        */
        const DoubleBuffer2D &getActionActivations() const {
            return _actionProbabilities;
        }

        /*!
        \brief Get the actions
        */
        const DoubleBuffer2D &getActions() const {
            return _actionTaken;
        }

        /*!
        \brief Get the actions in one-hot form
        */
        const cl::Image2D &getOneHotActions() const {
            return _oneHotAction;
        }

        /*!
        \brief Get number of action tiles in X and Y
        */
        cl_int2 getNumActionTiles() const {
            return _numActionTiles;
        }

        /*!
        \brief Get size of action tiles in X and Y
        */
        cl_int2 getActionTileSize() const {
            return _actionTileSize;
        }

        /*!
        \brief Get the hidden size
        */
        cl_int2 getHiddenSize() const {
            return _hiddenSize;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::AgentLayer* fbAgentLayer, ComputeSystem &cs);
        flatbuffers::Offset<schemas::AgentLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };
}