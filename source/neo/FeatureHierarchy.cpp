// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "FeatureHierarchy.h"

using namespace ogmaneo;

void FeatureHierarchy::createRandom(ComputeSystem &cs, ComputeProgram &program,
    const std::vector<InputDesc> &inputDescs, const std::vector<LayerDesc> &layerDescs,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
{
    _layerDescs = layerDescs;

    _layerDescs.front()._inputDescs = inputDescs;

    _layers.resize(_layerDescs.size());

    cl_int2 prevLayerSize = inputDescs.front()._size;

    for (int l = 0; l < _layers.size(); l++) {
        if (l == 0) {
            std::vector<SparseFeatures::VisibleLayerDesc> spDescs(inputDescs.size());

            for (int i = 0; i < inputDescs.size(); i++) {
                // Feed forward
                spDescs[i]._size = inputDescs[i]._size;
                spDescs[i]._radius = inputDescs[i]._radius;
                spDescs[i]._ignoreMiddle = false;
                spDescs[i]._weightAlpha = _layerDescs[l]._spFeedForwardWeightAlpha;
            }

            // Recurrent     
            if (_layerDescs[l]._recurrentRadius != 0) {
                SparseFeatures::VisibleLayerDesc recDesc;

                recDesc._size = _layerDescs[l]._size;
                recDesc._radius = _layerDescs[l]._recurrentRadius;
                recDesc._ignoreMiddle = true;
                recDesc._weightAlpha = _layerDescs[l]._spRecurrentWeightAlpha;

                spDescs.push_back(recDesc);
            }

            _layers[l]._sp.createRandom(cs, program, spDescs, _layerDescs[l]._size, _layerDescs[l]._inhibitionRadius, initWeightRange, rng);
        }
        else {
            std::vector<SparseFeatures::VisibleLayerDesc> spDescs(_layerDescs[l]._recurrentRadius != 0 ? 2 : 1);

            // Feed forward
            spDescs[0]._size = prevLayerSize;
            spDescs[0]._radius = _layerDescs[l]._inputDescs.front()._radius;
            spDescs[0]._ignoreMiddle = false;
            spDescs[0]._weightAlpha = _layerDescs[l]._spFeedForwardWeightAlpha;

            // Recurrent     
            if (_layerDescs[l]._recurrentRadius != 0) {
                spDescs[1]._size = _layerDescs[l]._size;
                spDescs[1]._radius = _layerDescs[l]._recurrentRadius;
                spDescs[1]._ignoreMiddle = true;
                spDescs[1]._weightAlpha = _layerDescs[l]._spRecurrentWeightAlpha;
            }

            _layers[l]._sp.createRandom(cs, program, spDescs, _layerDescs[l]._size, _layerDescs[l]._inhibitionRadius, initWeightRange, rng);
        }

        // Next layer
        prevLayerSize = _layerDescs[l]._size;
    }
}

void FeatureHierarchy::simStep(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, std::mt19937 &rng, bool learn) {
    std::vector<cl::Image2D> inputsUse = inputs;

    if (_layerDescs.front()._recurrentRadius != 0)
        inputsUse.push_back(_layers.front()._sp.getHiddenStates()[_back]);

    // Activate
    for (int l = 0; l < _layers.size(); l++) {
        std::vector<cl::Image2D> visibleStates = (l == 0) ? inputsUse : (_layerDescs[l]._recurrentRadius != 0 ? std::vector<cl::Image2D>{ _layers[l - 1]._sp.getHiddenStates()[_front], _layers[l]._sp.getHiddenStates()[_back] } : std::vector<cl::Image2D>{ _layers[l - 1]._sp.getHiddenStates()[_front] });

        _layers[l]._sp.activate(cs, visibleStates, _layerDescs[l]._spActiveRatio, rng);
    }

    // Learn
    for (int l = 0; l < _layers.size(); l++)
        _layers[l]._sp.learn(cs, _layerDescs[l]._spBiasAlpha, _layerDescs[l]._spActiveRatio);

    // Step end
    for (int l = 0; l < _layers.size(); l++)
        _layers[l]._sp.stepEnd(cs);
}

void FeatureHierarchy::clearMemory(ComputeSystem &cs) {
    for (int l = 0; l < _layers.size(); l++)
        _layers[l]._sp.clearMemory(cs);
}

void FeatureHierarchy::InputDesc::load(const schemas::hierarchy::InputDesc* inputDesc) {
    _size = cl_int2{ inputDesc->_size().x(), inputDesc->_size().y() };
    _radius = inputDesc->_radius();
}

void FeatureHierarchy::LayerDesc::load(const schemas::hierarchy::LayerDesc* fbLayerDesc) {
    if (!_inputDescs.empty()) {
        assert(_inputDescs.size() == fbLayerDesc->_inputDescs()->Length());
    }
    else {
        _inputDescs.reserve(fbLayerDesc->_inputDescs()->Length());
    }

    _size = cl_int2{ fbLayerDesc->_size()->x(), fbLayerDesc->_size()->y() };

    for (flatbuffers::uoffset_t i = 0; i < fbLayerDesc->_inputDescs()->Length(); i++) {
        _inputDescs[i].load(fbLayerDesc->_inputDescs()->Get(i));
    }

    _recurrentRadius = fbLayerDesc->_recurrentRadius();
    _inhibitionRadius = fbLayerDesc->_inhibitionRadius();
    _spFeedForwardWeightAlpha = fbLayerDesc->_spFeedForwardWeightAlpha();
    _spRecurrentWeightAlpha = fbLayerDesc->_spRecurrentWeightAlpha();
    _spBiasAlpha = fbLayerDesc->_spBiasAlpha();
    _spActiveRatio = fbLayerDesc->_spActiveRatio();
}

void FeatureHierarchy::Layer::load(const schemas::hierarchy::Layer* fbLayer, ComputeSystem &cs) {
    _sp.load(fbLayer->_sp(), cs);
}

void FeatureHierarchy::load(const schemas::hierarchy::FeatureHierarchy* fbFH, ComputeSystem &cs) {
    // Loading into an existing FeatureHierarchy?
    if (!_layerDescs.empty()) {
        assert(_layerDescs.size() == fbFH->_layerDescs()->Length());
        assert(_layers.size() == fbFH->_layers()->Length());
    }
    else {
        _layerDescs.reserve(fbFH->_layerDescs()->Length());
        _layers.reserve(fbFH->_layers()->Length());
    }

    for (flatbuffers::uoffset_t i = 0; i < fbFH->_layerDescs()->Length(); i++) {
        _layerDescs[i].load(fbFH->_layerDescs()->Get(i));
    }

    for (flatbuffers::uoffset_t i = 0; i < fbFH->_layers()->Length(); i++) {
        _layers[i].load(fbFH->_layers()->Get(i), cs);
    }

}

flatbuffers::Offset<schemas::hierarchy::LayerDesc> FeatureHierarchy::LayerDesc::save(flatbuffers::FlatBufferBuilder& builder) {
    schemas::int2 size(_size.x, _size.y);
    
    std::vector<schemas::hierarchy::InputDesc> inputDescs;

    for (InputDesc inputDesc : _inputDescs)
        inputDescs.push_back(inputDesc.save(builder));

    return schemas::hierarchy::CreateLayerDesc(builder,
        &size, builder.CreateVectorOfStructs(inputDescs), _recurrentRadius, _inhibitionRadius,
        _spFeedForwardWeightAlpha, _spRecurrentWeightAlpha,
        _spBiasAlpha, _spActiveRatio
    );
}

schemas::hierarchy::InputDesc FeatureHierarchy::InputDesc::save(flatbuffers::FlatBufferBuilder &builder) {
    schemas::int2 size(_size.x, _size.y);
   
    schemas::hierarchy::InputDesc inputDesc(
        size, _radius
    );

    return inputDesc;
}

flatbuffers::Offset<schemas::hierarchy::Layer> FeatureHierarchy::Layer::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    return schemas::hierarchy::CreateLayer(builder, _sp.save(builder, cs));
}

flatbuffers::Offset<schemas::hierarchy::FeatureHierarchy> FeatureHierarchy::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::hierarchy::LayerDesc>> layerDescs;

    for (LayerDesc layerDesc : _layerDescs)
        layerDescs.push_back(layerDesc.save(builder));

    std::vector<flatbuffers::Offset<schemas::hierarchy::Layer>> layers;

    for (Layer layer : _layers)
        layers.push_back(layer.save(builder, cs));

    // Build the FeatureHierarchy
    flatbuffers::Offset<schemas::hierarchy::FeatureHierarchy> ph = schemas::hierarchy::CreateFeatureHierarchy(
        builder,
        builder.CreateVector(layers),
        builder.CreateVector(layerDescs)
    );

    return ph;
}