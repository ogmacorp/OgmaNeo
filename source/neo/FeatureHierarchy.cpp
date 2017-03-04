// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "FeatureHierarchy.h"
#include "SparseFeaturesChunk.h"
#include "PredictorLayer.h"

using namespace ogmaneo;

void FeatureHierarchy::createRandom(ComputeSystem &cs, ComputeProgram &fhProgram,
    const std::vector<LayerDesc> &layerDescs,
    std::mt19937 &rng)
{
    _layerDescs = layerDescs;

    _layers.resize(_layerDescs.size());

    for (int l = 0; l < _layers.size(); l++)
        _layers[l]._sf = _layerDescs[l]._sfDesc->sparseFeaturesFactory();
}

void FeatureHierarchy::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, std::mt19937 &rng) {
    // Add a sample to the first layer
    _layers.front()._sf->subSample(cs, inputs, rng);

    // Activate
    bool prevClockReset = true;

    for (int l = 0; l < _layers.size(); l++) {
        // Add input to pool
        if (prevClockReset) {
            _layers[l]._clock++;

            // Update layer
            _layers[l]._sf->activate(cs, rng);

            _layers[l]._sf->stepEnd(cs);

            // Add a sample to the next layer
            if (l < _layers.size() - 1)
                _layers[l + 1]._sf->subSample(cs, { _layers[l]._sf->getHiddenStates()[_back] }, rng);
        }

        _layers[l]._tpReset = prevClockReset;

        if (_layers[l]._clock >= _layerDescs[l]._poolSteps) {
            _layers[l]._clock = 0;

            prevClockReset = true;
        }
        else
            prevClockReset = false;

        _layers[l]._tpNextReset = prevClockReset;
    }
}

void FeatureHierarchy::learn(ComputeSystem &cs, std::mt19937 &rng) {
    for (int l = 0; l < _layers.size(); l++) {
        // Add input to pool
        if (_layers[l]._tpReset)
            _layers[l]._sf->learn(cs, rng);
    }
}

void FeatureHierarchy::clearMemory(ComputeSystem &cs) {
    for (int l = 0; l < _layers.size(); l++)
        _layers[l]._sf->clearMemory(cs);
}

void FeatureHierarchy::LayerDesc::load(const schemas::FeatureHierarchyLayerDesc* fbFeatureHierarchyLayerDesc, ComputeSystem &cs) {
    _sfDesc->load(fbFeatureHierarchyLayerDesc->_sfDesc(), cs);
    _poolSteps = fbFeatureHierarchyLayerDesc->_poolSteps();
}

flatbuffers::Offset<schemas::FeatureHierarchyLayerDesc> FeatureHierarchy::LayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    return schemas::CreateFeatureHierarchyLayerDesc(builder,
        _sfDesc->save(builder, cs), _poolSteps);
}

void FeatureHierarchy::Layer::load(const schemas::FeatureHierarchyLayer* fbFeatureHierarchyLayer, ComputeSystem &cs) {
    schemas::SparseFeatures* fbSparseFeatures =
        (schemas::SparseFeatures*)(fbFeatureHierarchyLayer->_sf());
    _sf->load(fbSparseFeatures, cs);

    _clock = fbFeatureHierarchyLayer->_clock();
    _tpReset = fbFeatureHierarchyLayer->_tpReset();
    _tpNextReset = fbFeatureHierarchyLayer->_tpNextReset();
}

flatbuffers::Offset<schemas::FeatureHierarchyLayer> FeatureHierarchy::Layer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::SparseFeaturesType type;
    switch (_sf->_type) {
    default:
    case SparseFeaturesType::_chunk:    type = schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesChunk; break;
    case SparseFeaturesType::_distance: type = schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesDistance; break;
    }

    return schemas::CreateFeatureHierarchyLayer(builder,
        type, _sf->save(builder, cs).Union(),
        _clock,
        _tpReset, _tpNextReset);
}

void FeatureHierarchy::load(const schemas::FeatureHierarchy* fbFeatureHierarchy, ComputeSystem &cs) {
    assert(_layerDescs.size() == fbFeatureHierarchy->_layerDescs()->Length());
    assert(_layers.size() == fbFeatureHierarchy->_layers()->Length());

    for (flatbuffers::uoffset_t i = 0; i < fbFeatureHierarchy->_layerDescs()->Length(); i++) {
        _layerDescs[i].load(fbFeatureHierarchy->_layerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbFeatureHierarchy->_layers()->Length(); i++) {
        _layers[i].load(fbFeatureHierarchy->_layers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::FeatureHierarchy> FeatureHierarchy::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::FeatureHierarchyLayerDesc>> layerDescs;
    for (LayerDesc layerDesc : _layerDescs)
        layerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::FeatureHierarchyLayer>> layers;
    for (Layer layer : _layers)
        layers.push_back(layer.save(builder, cs));

    return schemas::CreateFeatureHierarchy(builder,
        builder.CreateVector(layerDescs), builder.CreateVector(layers));
}