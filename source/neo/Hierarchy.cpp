// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <assert.h>

using namespace ogmaneo;

Hierarchy::Hierarchy(ComputeSystem &cs, ComputeProgram &prog,
    int inputWidth, int inputHeight,
    const std::vector<LayerDescs> &layerDescs,
    float initMinWeight, float initMaxWeight, int seed)
{
    _pCs = &cs;
    _rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;

    cl_int2 inputSize = { inputWidth, inputHeight };

    std::vector<Predictor::PredLayerDesc> pLayerDescs(layerDescs.size());
    std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        hLayerDescs[l]._size = cl_int2{ layerDescs[l]._width, layerDescs[l]._height };
        hLayerDescs[l]._inputDescs = { FeatureHierarchy::InputDesc(inputSize, layerDescs[l]._feedForwardRadius) };
        hLayerDescs[l]._recurrentRadius = layerDescs[l]._recurrentRadius;
        hLayerDescs[l]._inhibitionRadius = layerDescs[l]._inhibitionRadius;
        hLayerDescs[l]._spFeedForwardWeightAlpha = layerDescs[l]._spFeedForwardWeightAlpha;
        hLayerDescs[l]._spRecurrentWeightAlpha = layerDescs[l]._spRecurrentWeightAlpha;
        hLayerDescs[l]._spBiasAlpha = layerDescs[l]._spBiasAlpha;
        hLayerDescs[l]._spActiveRatio = layerDescs[l]._spActiveRatio;

        pLayerDescs[l]._radius = layerDescs[l]._predRadius;
        pLayerDescs[l]._alpha = layerDescs[l]._predAlpha;
        pLayerDescs[l]._beta = layerDescs[l]._predBeta;
    }

    _ph.createRandom(cs, prog, inputSize, pLayerDescs, hLayerDescs, cl_float2{ initMinWeight, initMaxWeight }, _rng);

    // Create temporary buffers
    _inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputWidth, inputHeight);

    _pred.clear();
    _pred.assign(inputWidth * inputHeight, 0.0f);
}

void Hierarchy::simStep(const std::vector<float> &inputs, bool learn) {
    assert(inputs.size() == _inputWidth * _inputHeight);

    std::vector<float> inputsf = inputs;

    // Write input
    _pCs->getQueue().enqueueWriteImage(_inputImage,
        CL_TRUE, { 0, 0, 0 },
        { static_cast<cl::size_type>(_inputWidth), static_cast<cl::size_type>(_inputHeight), 1 },
        0, 0, inputsf.data());

    _ph.simStep(*_pCs, _inputImage, _inputImage, _rng, learn);

    // Get prediction
    _pCs->getQueue().enqueueReadImage(_ph.getPrediction(),
        CL_TRUE, { 0, 0, 0 },
        { static_cast<cl::size_type>(_inputWidth), static_cast<cl::size_type>(_inputHeight), 1 },
        0, 0, _pred.data());
}

void Hierarchy::load(const schemas::hierarchy::Hierarchy* fbH, ComputeSystem &cs) {
    _ph.load(fbH->_ph(), cs);

    _inputWidth = fbH->_inputWidth();
    _inputHeight = fbH->_inputHeight();

    ogmaneo::load(_inputImage, fbH->_inputImage(), cs);

    _pred.clear();
    for (flatbuffers::uoffset_t i = 0; i < fbH->_pred()->Length(); i++) {
        _pred.push_back(fbH->_pred()->Get(i));
    }

}

flatbuffers::Offset<schemas::hierarchy::Hierarchy> Hierarchy::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    return schemas::hierarchy::CreateHierarchy(builder,
        _ph.save(builder, cs),
        _inputWidth, _inputHeight,
        ogmaneo::save(_inputImage, builder, cs),
        builder.CreateVector(_pred));
}

void Hierarchy::load(ComputeSystem &cs, ComputeProgram &prog, const std::string &fileName) {
    FILE* file = fopen(fileName.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0L, SEEK_SET);
    std::vector<uint8_t> data(length);
    fread(data.data(), sizeof(uint8_t), length, file);
    fclose(file);

    flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);

    bool verified =
        schemas::hierarchy::VerifyHierarchyBuffer(verifier) |
        schemas::hierarchy::HierarchyBufferHasIdentifier(data.data());

    if (verified) {
        const schemas::hierarchy::Hierarchy* hierarchy = schemas::hierarchy::GetHierarchy(data.data());

        load(hierarchy, cs);
    }

    return; //verified;
}

void Hierarchy::save(ComputeSystem &cs, const std::string &fileName) {
    flatbuffers::FlatBufferBuilder builder;

    flatbuffers::Offset<schemas::hierarchy::Hierarchy> hierarchy = save(builder, cs);

    // Instruct the builder that this Hierarchy is complete.
    schemas::hierarchy::FinishHierarchyBuffer(builder, hierarchy);

    // Get the built agent and size
    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier = flatbuffers::Verifier(buf, size);

    bool verified =
        schemas::hierarchy::VerifyHierarchyBuffer(verifier) |
        schemas::hierarchy::HierarchyBufferHasIdentifier(buf);

    if (verified) {
        FILE* file = fopen(fileName.c_str(), "wb");
        fwrite(buf, sizeof(uint8_t), size, file);
        fclose(file);
    }

    return; //verified;
}