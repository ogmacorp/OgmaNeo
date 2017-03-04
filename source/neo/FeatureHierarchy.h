// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "SparseFeatures.h"
#include "schemas/FeatureHierarchy_generated.h"

namespace ogmaneo {
    /*!
    \brief Hierarchy of sparse features
    */
    class OGMA_API FeatureHierarchy {
    public:
        /*!
        \brief Layer desc
        Descriptor of a layer in the feature hierarchy
        */
        struct LayerDesc {
            /*!
            \brief Sparse features desc
            */
            std::shared_ptr<SparseFeatures::SparseFeaturesDesc> _sfDesc;

            /*!
            \brief Stride length
            */
            int _poolSteps;

            /*!
            \brief Initialize defaults
            */
            LayerDesc()
                : _poolSteps(2)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::FeatureHierarchyLayerDesc* fbFeatureHierarchyLayerDesc, ComputeSystem &cs);
            flatbuffers::Offset<schemas::FeatureHierarchyLayerDesc> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

        /*!
        \brief Layer
        */
        struct Layer {
            /*!
            \brief Sparse features
            */
            std::shared_ptr<SparseFeatures> _sf;

            /*!
            \brief Clock for striding (relative to previous layer)
            */
            int _clock;

            //!@{
            /*!
            \brief Flags for use by other systems
            */
            bool _tpReset;
            bool _tpNextReset;
            //!@}

            /*!
            \brief Initialize defaults
            */
            Layer()
                : _clock(0), _tpReset(false), _tpNextReset(false)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::FeatureHierarchyLayer* fbFeatureHierarchyLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::FeatureHierarchyLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
            //!@}
        };

    private:
        //!@{
        /*!
        \brief Layers and descs
        */
        std::vector<Layer> _layers;
        std::vector<LayerDesc> _layerDescs;
        //!@}

    public:
        /*!
        \brief Initialize defaults
        */
        FeatureHierarchy()
        {}

        /*!
        \brief Create a sparse feature hierarchy with random initialization
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &fhProgram,
            const std::vector<LayerDesc> &layerDescs,
            std::mt19937 &rng);

        /*!
        \brief Activation of the hierarchy
        Runs one timestep of activation (no learning).
        \param inputs the inputs to the bottom-most layer.
        \param rng a random number generator.
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, std::mt19937 &rng);

        /*!
        \brief Learn step of hierarchy
        Runs one timestep of learning. Typically called after a call to activate(...).
        \param inputs the inputs to the bottom-most layer.
        \param rng a random number generator.
        */
        void learn(ComputeSystem &cs, std::mt19937 &rng);

        /*!
        \brief Get number of layers
        */
        size_t getNumLayers() const {
            return _layers.size();
        }

        /*!
        \brief Get access to a layer
        \param index index Layer index.
        */
        const Layer &getLayer(int index) const {
            return _layers[index];
        }

        /*!
        \brief Get access to a layer desc
        \param index index Layer index.
        */
        const LayerDesc &getLayerDesc(int index) const {
            return _layerDescs[index];
        }

        /*!
        \brief Clear the working memory
        \param cs is the ComputeSystem.
        */
        void clearMemory(ComputeSystem &cs);

        //!@{
        /*!
        \brief Serialization
        */
        void load(const schemas::FeatureHierarchy* fbFeatureHierarchy, ComputeSystem &cs);
        flatbuffers::Offset<schemas::FeatureHierarchy> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };
}