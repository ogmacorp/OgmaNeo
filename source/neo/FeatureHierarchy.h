// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "OgmaNeo.h"
#include "SparseFeatures.h"
#include "ImageWhitener.h"
#include "schemas/FeatureHierarchy_generated.h"

namespace ogmaneo {
    /*!
    \brief Hierarchy of sparse features
    */
    class OGMA_API FeatureHierarchy {
    public:
        /*!
        \brief Input desc
        Descriptor of a layer input
        */
        struct InputDesc {
            /*!
            \brief Size of input layer
            */
            cl_int2 _size;

            /*!
            \brief Radii for feed forward and inhibitory connections
            */
            cl_int _radius;

            /*!
            \brief Initialize defaults
            */
            InputDesc()
            {}

            /*!
            \brief Initialize from values
            */
            InputDesc(cl_int2 size, cl_int radius)
                : _size(size), _radius(radius)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::hierarchy::InputDesc* inputDesc);
            schemas::hierarchy::InputDesc save(flatbuffers::FlatBufferBuilder &builder);
            //!@}
        };

        /*!
        \brief Layer desc
        Descriptor of a layer in the feature hierarchy
        */
        struct LayerDesc {
            /*!
            \brief Size of layer
            */
            cl_int2 _size;

            /*!
            \brief Input descriptors
            */
            std::vector<InputDesc> _inputDescs;

            /*!
            \brief Radius for recurrent connections
            */
            cl_int _recurrentRadius;

            /*!
            \brief Radius for inhibitory connections
            */
            cl_int _inhibitionRadius;

            //!@{
            /*!
            \brief Sparse predictor parameters
            */
            cl_float _spFeedForwardWeightAlpha;
            cl_float _spRecurrentWeightAlpha;
            cl_float _spBiasAlpha;
            cl_float _spActiveRatio;
            //!@}

            /*!
            \brief Initialize defaults
            */
            LayerDesc()
                : _size({ 8, 8 }),
                _inputDescs({ InputDesc({ 16, 16 }, 6) }), _recurrentRadius(6), _inhibitionRadius(5),
                _spFeedForwardWeightAlpha(0.25f), _spRecurrentWeightAlpha(0.25f), _spBiasAlpha(0.01f),
                _spActiveRatio(0.02f)
            {}

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::hierarchy::LayerDesc* fbLayerDesc);
            flatbuffers::Offset<schemas::hierarchy::LayerDesc> save(flatbuffers::FlatBufferBuilder &builder);
            //!@}
        };

        /*!
        \brief Layer
        */
        struct Layer {
            /*!
            \brief Sparse predictor
            */
            SparseFeatures _sp;

            //!@{
            /*!
            \brief Serialization
            */
            void load(const schemas::hierarchy::Layer* fbLayer, ComputeSystem &cs);
            flatbuffers::Offset<schemas::hierarchy::Layer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
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
        \brief Create a sparse feature hierarchy with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OmgaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param inputDescs are the descriptors of the input layers.
        \param layerDescs are descriptors for feature hierarchy layers.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &program,
            const std::vector<InputDesc> &inputDescs, const std::vector<LayerDesc> &layerDescs,
            cl_float2 initWeightRange,
            std::mt19937 &rng);

        /*!
        \brief Simulation step of hierarchy
        Runs one timestep of simulation.
        \param cs is the ComputeSystem.
        \param inputs the inputs to the bottom-most layer.
        \param rng a random number generator.
        \param learn optional argument to disable learning.
        */
        void simStep(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, std::mt19937 &rng, bool learn = true);

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
        void load(const schemas::hierarchy::FeatureHierarchy* fbFH, ComputeSystem &cs);
        flatbuffers::Offset<schemas::hierarchy::FeatureHierarchy> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };
}