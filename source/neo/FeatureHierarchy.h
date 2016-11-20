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
            \brief Temporal pooling
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
            \brief Clock for temporal pooling (relative to previous layer)
            */
            int _clock;

            /*!
            \brief Temporal pooling buffer
            */
            DoubleBuffer2D _tpBuffer;

            /*!
            \brief Prediction error temporary buffer
            */
            cl::Image2D _predErrors;

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

        //!@{
        /*!
        \brief Additional kernels
        */
        cl::Kernel _fhPoolKernel;
        cl::Kernel _fhPredErrorKernel;
        //!@}

    public:
        /*!
        \brief Initialize defaults
        */
        FeatureHierarchy()
        {}

        /*!
        \brief Create a sparse feature hierarchy with random initialization
        Requires the compute system, program with the NeoRL kernels, and initialization information.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &fhProgram,
            const std::vector<LayerDesc> &layerDescs,
            std::mt19937 &rng);

        /*!
        \brief Simulation step of hierarchy
        Runs one timestep of simulation.
        \param inputs the inputs to the bottom-most layer.
        \param rng a random number generator.
        \param learn optionally disable learning
        */
        void simStep(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, const std::vector<cl::Image2D> &predictionsPrev, std::mt19937 &rng, bool learn = true);

        /*!
        \brief Get number of layers
        */
        size_t getNumLayers() const {
            return _layers.size();
        }

        /*!
        \brief Get access to a layer
        \param[in] index Layer index.
        */
        const Layer &getLayer(int index) const {
            return _layers[index];
        }

        /*!
        \brief Get access to a layer desc
        \param[in] index Layer index.
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