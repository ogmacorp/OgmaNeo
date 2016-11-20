// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "Helpers.h"
#include "schemas/SparseFeatures_generated.h"

namespace ogmaneo {
    /*!
    \brief Sparse Features
    Base class for encoders (sparse features)
    */
    class OGMA_API SparseFeatures {
    public:
        enum OGMA_API InputType {
            _feedForward, _feedForwardRecurrent
        };

        SparseFeaturesType _type;

    public:
        /*!
        \brief Sparse Features Descriptor
        Base class for encoder descriptors (sparse features descriptors)
        */
        class OGMA_API SparseFeaturesDesc {
        public:
            std::string _name;

            InputType _inputType;

            virtual size_t getNumVisibleLayers() const = 0;
            virtual cl_int2 getVisibleLayerSize(int vli) const = 0;

            virtual cl_int2 getHiddenSize() const = 0;

            virtual std::shared_ptr<SparseFeatures> sparseFeaturesFactory() = 0;

            /*!
            \brief Initialize defaults
            */
            SparseFeaturesDesc()
                : _name("Unassigned"), _inputType(_feedForward)
            {}

            virtual ~SparseFeaturesDesc() {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::SparseFeaturesDesc* fbSparseFeaturesDesc, ComputeSystem &cs) {
                _name = fbSparseFeaturesDesc->_name()->c_str();

                switch (fbSparseFeaturesDesc->_inputType()) {
                default:
                case schemas::InputType::InputType__feedForward:            _inputType = InputType::_feedForward; break;
                case schemas::InputType::InputType__feedForwardRecurrent:   _inputType = InputType::_feedForwardRecurrent; break;
                }
            }

            flatbuffers::Offset<schemas::SparseFeaturesDesc> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
                schemas::InputType inputType;
                switch (_inputType) {
                default:
                case InputType::_feedForward:           inputType = schemas::InputType::InputType__feedForward; break;
                case InputType::_feedForwardRecurrent:  inputType = schemas::InputType::InputType__feedForwardRecurrent; break;
                }

                return schemas::CreateSparseFeaturesDesc(builder,
                    builder.CreateString(_name), inputType);
            }
            //!@}
        };

        virtual ~SparseFeatures() {}

        /*!
        \brief Activate predictor
        \param cs is the ComputeSystem.
        \param visibleStates the input layer states.
        \param lambda decay of hidden unit traces.
        \param activeRatio % active units.
        \param rng a random number generator.
        */
        virtual void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &predictionsPrev, std::mt19937 &rng) = 0;

        /*!
        \brief End a simulation step
        */
        virtual void stepEnd(ComputeSystem &cs) = 0;

        /*!
        \brief Learning
        */
        virtual void learn(ComputeSystem &cs, std::mt19937 &rng) = 0;

        /*!
        \brief Inhibition
        */
        virtual void inhibit(ComputeSystem &cs, const cl::Image2D &activations, cl::Image2D &states, std::mt19937 &rng) = 0;

        /*!
        \brief Get hidden size
        */
        virtual cl_int2 getHiddenSize() const = 0;

        /*!
        \brief Get hidden states
        */
        virtual const DoubleBuffer2D &getHiddenStates() const = 0;

        /*!
        \brief Get context
        */
        virtual const cl::Image2D &getHiddenContext() const {
            return getHiddenStates()[_back];
        }

        /*!
        \brief Clear the working memory
        */
        virtual void clearMemory(ComputeSystem &cs) = 0;

        //!@{
        /*!
        \brief Serialization
        */
        virtual void load(const schemas::SparseFeatures* fbSparseFeatures, ComputeSystem &cs) = 0;
        virtual flatbuffers::Offset<schemas::SparseFeatures> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) = 0;
        //!@}
    };
}