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

void Predictor::createRandom(ComputeSystem &cs, ComputeProgram &program,
    cl_int2 inputSize, const std::vector<PredLayerDesc> &pLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
    cl_float2 initWeightRange,
    std::mt19937 &rng, float firstLearningRateScalar)
{
    assert(pLayerDescs.size() > 0);
    assert(pLayerDescs.size() == hLayerDescs.size());

    _inputSize = inputSize;

    // Create underlying hierarchy
    _h.createRandom(cs, program, std::vector<FeatureHierarchy::InputDesc>{ FeatureHierarchy::InputDesc(inputSize, hLayerDescs.front()._inputDescs.front()._radius) }, hLayerDescs, initWeightRange, rng);

    cl_int2 prevLayerSize = inputSize;

    _pLayerDescs = pLayerDescs;

    _pLayers.resize(_pLayerDescs.size());

    for (int l = 0; l < _pLayers.size(); l++) {
        std::vector<PredictorLayer::VisibleLayerDesc> pVisibleLayerDescs;

        float alpha = (l == 0) ? firstLearningRateScalar * _pLayerDescs[l]._alpha : _pLayerDescs[l]._alpha;
        float beta = (l == 0) ? firstLearningRateScalar * _pLayerDescs[l]._beta : _pLayerDescs[l]._beta;

        if (l < _pLayers.size() - 1) {
            pVisibleLayerDescs.resize(2);

            // Current
            pVisibleLayerDescs[0]._radius = _pLayerDescs[l]._radius;
            pVisibleLayerDescs[0]._alpha = alpha;
            pVisibleLayerDescs[0]._size = hLayerDescs[l]._size;

            // Feed back
            pVisibleLayerDescs[1]._radius = _pLayerDescs[l]._radius;
            pVisibleLayerDescs[1]._alpha = beta;
            pVisibleLayerDescs[1]._size = hLayerDescs[l]._size;
        }
        else {
            pVisibleLayerDescs.resize(1);

            // Current
            pVisibleLayerDescs[0]._radius = _pLayerDescs[l]._radius;
            pVisibleLayerDescs[0]._alpha = alpha;
            pVisibleLayerDescs[0]._size = hLayerDescs[l]._size;
        }

        _pLayers[l].createRandom(cs, program, prevLayerSize, pVisibleLayerDescs, initWeightRange, rng);

        // Next layer
        prevLayerSize = hLayerDescs[l]._size;
    }
}

void Predictor::simStep(ComputeSystem &cs, const cl::Image2D &input, const cl::Image2D &inputCorrupted, std::mt19937 &rng, bool learn) {
    // Activate hierarchy
    _h.simStep(cs, { inputCorrupted }, rng, learn);

    // Forward pass through predictor to get next prediction
    for (int l = (int)_pLayers.size() - 1; l >= 0; l--) {
        cl::Image2D prevPredictions = (l == _pLayers.size() - 1) ? _h.getLayer(l)._sp.getHiddenStates()[_back] : _pLayers[l + 1].getHiddenStates()[_front];

        if (l < _pLayers.size() - 1)
            _pLayers[l].activate(cs, std::vector<cl::Image2D>{ _h.getLayer(l)._sp.getHiddenStates()[_back], prevPredictions }, l != 0);
        else
            _pLayers[l].activate(cs, std::vector<cl::Image2D>{ _h.getLayer(l)._sp.getHiddenStates()[_back] }, l != 0);
    }

    if (learn) {
        for (int l = (int)_pLayers.size() - 1; l >= 0; l--) {
            cl::Image2D target = l == 0 ? input : _h.getLayer(l - 1)._sp.getHiddenStates()[_back];

            cl::Image2D prevPredictions = (l == _pLayers.size() - 1) ? _h.getLayer(l)._sp.getHiddenStates()[_front] : _pLayers[l + 1].getHiddenStates()[_back];

            if (l < _pLayers.size() - 1)
                _pLayers[l].learn(cs, target, std::vector<cl::Image2D>{ _h.getLayer(l)._sp.getHiddenStates()[_front], prevPredictions });
            else
                _pLayers[l].learn(cs, target, std::vector<cl::Image2D>{ _h.getLayer(l)._sp.getHiddenStates()[_front] });       
        }
    }

    for (int l = 0; l < _pLayers.size(); l++)
        _pLayers[l].stepEnd(cs);
}

void Predictor::PredLayerDesc::load(const schemas::predictor::LayerDesc* fbLayerDesc) {
    _radius = fbLayerDesc->_radius();
    _alpha = fbLayerDesc->_alpha();
    _beta = fbLayerDesc->_beta();
}

void Predictor::load(const schemas::predictor::Predictor* fbPredictor, ComputeSystem &cs) {
    if (!_pLayers.empty()) {
        assert(_inputSize.x == fbPredictor->_inputSize()->x());
        assert(_inputSize.y == fbPredictor->_inputSize()->y());
        assert(_pLayerDescs.size() == fbPredictor->_pLayerDescs()->Length());
        assert(_pLayers.size() == fbPredictor->_pLayers()->Length());
    }
    else {
        _inputSize.x = fbPredictor->_inputSize()->x();
        _inputSize.y = fbPredictor->_inputSize()->y();
        _pLayerDescs.reserve(fbPredictor->_pLayerDescs()->Length());
        _pLayers.reserve(fbPredictor->_pLayers()->Length());
    }

    _h.load(fbPredictor->_h(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayerDescs()->Length(); i++) {
        _pLayerDescs[i].load(fbPredictor->_pLayerDescs()->Get(i));
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayers()->Length(); i++) {
        _pLayers[i].load(fbPredictor->_pLayers()->Get(i), cs);
    }

}

void Predictor::load(ComputeSystem &cs, ComputeProgram& prog, const std::string& fileName) {
    FILE* file = fopen(fileName.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0L, SEEK_SET);
    std::vector<uint8_t> data(length);
    fread(data.data(), sizeof(uint8_t), length, file);
    fclose(file);

    flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);
    bool verified =
        schemas::predictor::VerifyPredictorBuffer(verifier) |
        schemas::predictor::PredictorBufferHasIdentifier(data.data());

    if (verified) {
        const schemas::predictor::Predictor* nfh = schemas::predictor::GetPredictor(data.data());
        load(nfh, cs);
    }

    return; //verified;
}

schemas::predictor::LayerDesc Predictor::PredLayerDesc::save(flatbuffers::FlatBufferBuilder& builder) {
    schemas::predictor::LayerDesc layerDesc(_radius, _alpha, _beta);
    return layerDesc;
}

flatbuffers::Offset<schemas::predictor::Predictor> Predictor::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 inputSize(_inputSize.x, _inputSize.y);

    std::vector<schemas::predictor::LayerDesc> layerDescs;

    for (PredLayerDesc layerDesc : _pLayerDescs)
        layerDescs.push_back(layerDesc.save(builder));

    std::vector<flatbuffers::Offset<schemas::predictor::Layer>> layers;

    for (PredictorLayer layer : _pLayers)
        layers.push_back(layer.save(builder, cs));

    // Build the Predictor hierarchy
    flatbuffers::Offset<schemas::predictor::Predictor> fh = schemas::predictor::CreatePredictor(
        builder,
        _h.save(builder, cs),
        &inputSize,
        builder.CreateVector(layers),
        builder.CreateVectorOfStructs(layerDescs)
    );

    return fh;
}

void Predictor::save(ComputeSystem &cs, const std::string& fileName) {
    flatbuffers::FlatBufferBuilder builder;

    flatbuffers::Offset<schemas::predictor::Predictor> fh = save(builder, cs);

    // Instruct the builder that this Predictor hierarchy is complete.
    schemas::predictor::FinishPredictorBuffer(builder, fh);

    // Get the built hierarchy and size
    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier = flatbuffers::Verifier(buf, size);

    bool verified =
        schemas::predictor::VerifyPredictorBuffer(verifier) |
        schemas::predictor::PredictorBufferHasIdentifier(buf);

    if (verified) {
        FILE* file = fopen(fileName.c_str(), "wb");
        fwrite(buf, sizeof(uint8_t), size, file);
        fclose(file);
    }

    return; //verified;
}