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
    const std::vector<cl_int2> &inputSizes, const std::vector<cl_int2> &inputChunkSizes,
    const std::vector<std::vector<PredLayerDesc>> &pLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
{
    assert(pLayerDescs.size() > 0);
    assert(pLayerDescs.size() == hLayerDescs.size());

    // Create underlying hierarchy
    _h.createRandom(cs, hProgram, hLayerDescs, rng);

    _pLayerDescs = pLayerDescs;

    _pLayers.resize(_pLayerDescs.size());

    _needsUpdate.clear();
    _needsUpdate.assign(_pLayerDescs.size(), false);

    for (int l = 0; l < _pLayers.size(); l++) {
        _pLayers[l].resize(_pLayerDescs[l].size());

        // All other layers predict timesteps downward
        for (int k = 0; k < _pLayers[l].size(); k++) {
            std::vector<PredictorLayer::VisibleLayerDesc> pVisibleLayerDescs;

            if (l < _pLayers.size() - 1) {
                pVisibleLayerDescs.resize(2);

                pVisibleLayerDescs[0]._radius = _pLayerDescs[l][k]._radius;
                pVisibleLayerDescs[0]._alpha = _pLayerDescs[l][k]._alpha;
                pVisibleLayerDescs[0]._lambda = _pLayerDescs[l][k]._lambda;
                pVisibleLayerDescs[0]._gamma = _pLayerDescs[l][k]._gamma;
                pVisibleLayerDescs[0]._size = _h.getLayer(l)._sf->getHiddenSize();

                pVisibleLayerDescs[1]._radius = _pLayerDescs[l][k]._radius;
                pVisibleLayerDescs[1]._alpha = _pLayerDescs[l][k]._beta;
                pVisibleLayerDescs[1]._lambda = _pLayerDescs[l][k]._lambda;
                pVisibleLayerDescs[1]._gamma = _pLayerDescs[l][k]._gamma;
                pVisibleLayerDescs[1]._size = _h.getLayer(l)._sf->getHiddenSize();
            }
            else {
                pVisibleLayerDescs.resize(1);

                pVisibleLayerDescs[0]._radius = _pLayerDescs[l][k]._radius;
                pVisibleLayerDescs[0]._alpha = _pLayerDescs[l][k]._alpha;
                pVisibleLayerDescs[0]._lambda = _pLayerDescs[l][k]._lambda;
                pVisibleLayerDescs[0]._gamma = _pLayerDescs[l][k]._gamma;
                pVisibleLayerDescs[0]._size = _h.getLayer(l)._sf->getHiddenSize();
            }

            if (l == 0)
                _pLayers[l][k].createRandom(cs, pProgram, inputSizes[k], pVisibleLayerDescs, _pLayerDescs[l][k]._isQ ? PredictorLayer::_q : PredictorLayer::_none, inputChunkSizes[k], initWeightRange, rng);
            else
                _pLayers[l][k].createRandom(cs, pProgram, _h.getLayer(l - 1)._sf->getHiddenSize(), pVisibleLayerDescs, PredictorLayer::_inhibitBinary, _h.getLayer(l - 1)._sf->getChunkSize(), initWeightRange, rng);
        }
    }
}

void Predictor::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &inputsFeed, std::mt19937 &rng) {
    // Activate hierarchy
    _h.activate(cs, inputsFeed, rng);

    // Forward pass through predictor to get next prediction
    for (int l = _pLayers.size() - 1; l >= 0; l--) {
        if (_h.getLayer(l)._tpReset) {
            _needsUpdate[l] = true;

            // Others make corrections over multiple (destrided) timesteps
            if (l < _pLayers.size() - 1) {
                for (int k = 0; k < _pLayers[l].size(); k++)
                    _pLayers[l][k].activate(cs, std::vector<cl::Image2D>{ _h.getLayer(l)._sf->getHiddenStates()[_back], _pLayers[l + 1][_h.getLayerDesc(l)._poolSteps - 1 - _h.getLayer(l)._clock].getHiddenStates()[_back] }, rng);
            }
            else {
                for (int k = 0; k < _pLayers[l].size(); k++)
                    _pLayers[l][k].activate(cs, std::vector<cl::Image2D>{ _h.getLayer(l)._sf->getHiddenStates()[_back] }, rng);
            }

            for (int k = 0; k < _pLayers[l].size(); k++)
                _pLayers[l][k].stepEnd(cs);
        }
    }
}

void Predictor::learn(ComputeSystem &cs, const std::vector<cl::Image2D> &inputsPredict, std::mt19937 &rng, float tdError) {
    std::vector<cl::Image2D> predictionsPrev(_pLayers.size());

    for (int l = 0; l < predictionsPrev.size(); l++)
        predictionsPrev[l] = _pLayers[l][0].getHiddenStates()[_back];

    // Activate hierarchy
    _h.learn(cs, rng);

    for (int l = 0; l < _pLayers.size(); l++) {
        if ((l == 0 || _h.getLayer(l - 1)._clock == 1) && _needsUpdate[l]) { // == 1 ?
            if (l == 0) {
                for (int k = 0; k < _pLayers[l].size(); k++)
                    _pLayers[l][k].learn(cs, inputsPredict[k], true, tdError);
            }
            else {
                for (int k = 0; k < _pLayers[l].size(); k++)
                    _pLayers[l][k].learn(cs, _h.getLayer(l)._sf->getSubSampleAccum(cs, 0, k, rng), true);
            }

            _needsUpdate[l] = false;
        }
    }
}

void Predictor::PredLayerDesc::load(const schemas::PredLayerDesc* fbPredLayerDesc, ComputeSystem &cs) {
    _isQ = fbPredLayerDesc->_isQ();
    _radius = fbPredLayerDesc->_radius();
    _alpha = fbPredLayerDesc->_alpha();
    _beta = fbPredLayerDesc->_beta();
    _lambda = fbPredLayerDesc->_lambda();
    _gamma = fbPredLayerDesc->_gamma();
}

flatbuffers::Offset<schemas::PredLayerDesc> Predictor::PredLayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    return schemas::CreatePredLayerDesc(builder, _isQ, _radius, _alpha, _beta, _lambda, _gamma);
}

void Predictor::load(const schemas::Predictor* fbPredictor, ComputeSystem &cs) {
    assert(_pLayerDescs.size() == fbPredictor->_pLayerDescs()->Length());
    assert(_pLayers.size() == fbPredictor->_pLayers()->Length());

    _h.load(fbPredictor->_h(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayerDescs()->Length(); i++) {
        assert(_pLayerDescs[i].size() == fbPredictor->_pLayerDescs()->Get(i)->_pLayerDescs()->Length());

        for (flatbuffers::uoffset_t j = 0; j < fbPredictor->_pLayerDescs()->Get(i)->_pLayerDescs()->Length(); j++) {
            _pLayerDescs[i][j].load(fbPredictor->_pLayerDescs()->Get(i)->_pLayerDescs()->Get(j), cs);
        }
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayers()->Length(); i++) {
        assert(_pLayers[i].size() == fbPredictor->_pLayers()->Get(i)->_pLayers()->Length());

        for (flatbuffers::uoffset_t j = 0; j < fbPredictor->_pLayerDescs()->Get(i)->_pLayerDescs()->Length(); j++) {
            _pLayers[i][j].load(fbPredictor->_pLayers()->Get(i)->_pLayers()->Get(j), cs);
        }
    }
}

flatbuffers::Offset<schemas::Predictor> Predictor::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::PredLayerDescs>> predLayerDescs;
    for (std::vector<PredLayerDesc> layerDescs : _pLayerDescs) {
        std::vector<flatbuffers::Offset<schemas::PredLayerDesc>> predLayerDesc;
        for (PredLayerDesc layerDesc : layerDescs)
            predLayerDesc.push_back(layerDesc.save(builder, cs));

        predLayerDescs.push_back(schemas::CreatePredLayerDescs(builder, builder.CreateVector(predLayerDesc)));
    }

    std::vector<flatbuffers::Offset<schemas::PredictorLayers>> predictorLayers;
    for (std::vector<PredictorLayer> layers : _pLayers) {
        std::vector<flatbuffers::Offset<schemas::PredictorLayer>> predictorLayer;
        for (PredictorLayer layer : layers)
            predictorLayer.push_back(layer.save(builder, cs));

        predictorLayers.push_back(schemas::CreatePredictorLayers(builder, builder.CreateVector(predictorLayer)));
    }

    return schemas::CreatePredictor(builder,
        _h.save(builder, cs), builder.CreateVector(predLayerDescs), builder.CreateVector(predictorLayers));
}