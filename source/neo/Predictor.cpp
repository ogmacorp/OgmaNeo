// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Predictor.h"

#include <iostream>

using namespace ogmaneo;

void Predictor::createRandom(ComputeSystem &cs, ComputeProgram &hProgram, ComputeProgram &pProgram,
    const std::vector<PredLayerDesc> &pLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
{
    assert(pLayerDescs.size() > 0);
    assert(pLayerDescs.size() == hLayerDescs.size());

    // Create underlying hierarchy
    _h.createRandom(cs, hProgram, hLayerDescs, rng);

    _pLayerDescs = pLayerDescs;

    _pLayers.resize(_pLayerDescs.size());

    for (int l = 0; l < _pLayers.size(); l++) {
        std::vector<PredictorLayer::VisibleLayerDesc> pVisibleLayerDescs(l == _pLayers.size() - 1 ? 1 : 2);

        for (int p = 0; p < pVisibleLayerDescs.size(); p++) {
            // Current
            pVisibleLayerDescs[p]._radius = _pLayerDescs[l]._radius;
            pVisibleLayerDescs[p]._alpha = (p == 0) ? _pLayerDescs[l]._alpha : _pLayerDescs[l]._beta;
            pVisibleLayerDescs[p]._size = _h.getLayer(l)._sf->getHiddenSize();
        }

        _pLayers[l].createRandom(cs, pProgram, _h.getLayer(l)._sf->getHiddenSize(), pVisibleLayerDescs, _h.getLayer(l)._sf, initWeightRange, rng);
    }
}

void Predictor::simStep(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, const std::vector<cl::Image2D> &inputsCorrupted, std::mt19937 &rng, bool learn) {
    std::vector<cl::Image2D> predictionsPrev(_pLayers.size());

    for (int l = 0; l < predictionsPrev.size(); l++)
        predictionsPrev[l] = _pLayers[l].getHiddenStates()[_back];

    // Activate hierarchy
    _h.simStep(cs, inputsCorrupted, predictionsPrev, rng, learn);

    // Forward pass through predictor to get next prediction
    for (int l = static_cast<int>(_pLayers.size()) - 1; l >= 0; l--) {
        if (_h.getLayer(l)._tpReset || _h.getLayer(l)._tpNextReset) {
            cl::Image2D target = _h.getLayer(l)._sf->getHiddenStates()[_back];

            if (l != _pLayers.size() - 1) {
                _pLayers[l].activate(cs, std::vector<cl::Image2D>{ _h.getLayer(l)._sf->getHiddenStates()[_back], _pLayers[l + 1].getHiddenStates()[_back] }, rng);

                if (learn)
                    _pLayers[l].learn(cs, target);
            }
            else {
                _pLayers[l].activate(cs, std::vector<cl::Image2D>{ _h.getLayer(l)._sf->getHiddenStates()[_back] }, rng);

                if (learn)
                    _pLayers[l].learn(cs, target);
            }

            _pLayers[l].stepEnd(cs);
        }
    }
}

void Predictor::PredLayerDesc::load(const schemas::PredLayerDesc* fbPredLayerDesc, ComputeSystem &cs) {
    _radius = fbPredLayerDesc->_radius();
    _alpha = fbPredLayerDesc->_alpha();
    _beta = fbPredLayerDesc->_beta();
}

schemas::PredLayerDesc Predictor::PredLayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    return schemas::PredLayerDesc(_radius, _alpha, _beta);
}

void Predictor::load(const schemas::Predictor* fbPredictor, ComputeSystem &cs) {
    assert(_pLayerDescs.size() == fbPredictor->_pLayerDescs()->Length());
    assert(_pLayers.size() == fbPredictor->_pLayers()->Length());

    _h.load(fbPredictor->_h(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayerDescs()->Length(); i++) {
        _pLayerDescs[i].load(fbPredictor->_pLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayers()->Length(); i++) {
        _pLayers[i].load(fbPredictor->_pLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::Predictor> Predictor::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<schemas::PredLayerDesc> predLayerDescs;
    for (PredLayerDesc layerDesc : _pLayerDescs)
        predLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::PredictorLayer>> predLayers;
    for (PredictorLayer layer : _pLayers)
        predLayers.push_back(layer.save(builder, cs));

    return schemas::CreatePredictor(builder,
        _h.save(builder, cs), builder.CreateVectorOfStructs(predLayerDescs), builder.CreateVector(predLayers));
}