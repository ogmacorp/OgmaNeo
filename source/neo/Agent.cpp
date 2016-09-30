// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Agent.h"

#include <assert.h>

using namespace ogmaneo;

Agent::Agent(ComputeSystem &cs, ComputeProgram &prog,
    int inputWidth, int inputHeight,
    int actionWidth, int actionHeight,
    int actionTileWidth, int actionTileHeight,
    int actionRadius,
    const std::vector<LayerDescs> &layerDescs,
    float initMinWeight, float initMaxWeight, int seed)
{
    _pCs = &cs;
    _rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;
    _actionWidth = actionWidth;
    _actionHeight = actionHeight;
    _actionTileWidth = actionTileWidth;
    _actionTileHeight = actionTileHeight;

    cl_int2 inputSize = { inputWidth, inputHeight };
    cl_int2 actionSize = { actionWidth, actionHeight };

    std::vector<AgentSwarm::AgentLayerDesc> aLayerDescs(layerDescs.size());
    std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        hLayerDescs[l]._size = cl_int2{ layerDescs[l]._width, layerDescs[l]._height };
        hLayerDescs[l]._inputDescs = { FeatureHierarchy::InputDesc(inputSize, layerDescs[l]._feedForwardRadius) };
        hLayerDescs[l]._inhibitionRadius = layerDescs[l]._inhibitionRadius;
        hLayerDescs[l]._recurrentRadius = layerDescs[l]._recurrentRadius;
        hLayerDescs[l]._spFeedForwardWeightAlpha = layerDescs[l]._spFeedForwardWeightAlpha;
        hLayerDescs[l]._spRecurrentWeightAlpha = layerDescs[l]._spRecurrentWeightAlpha;
        hLayerDescs[l]._spBiasAlpha = layerDescs[l]._spBiasAlpha;
        hLayerDescs[l]._spActiveRatio = layerDescs[l]._spActiveRatio;

        aLayerDescs[l]._radius = layerDescs[l]._qRadius;     
        aLayerDescs[l]._qAlpha = layerDescs[l]._qAlpha;
        aLayerDescs[l]._qGamma = layerDescs[l]._qGamma;
        aLayerDescs[l]._qLambda = layerDescs[l]._qLambda;
        aLayerDescs[l]._epsilon = layerDescs[l]._epsilon;
    }

    _as.createRandom(cs, prog, inputSize, actionSize, { actionTileWidth, actionTileHeight }, actionRadius, aLayerDescs, hLayerDescs, cl_float2{ initMinWeight, initMaxWeight }, _rng);

    // Create temporary buffers
    _inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), inputWidth, inputHeight);

    _action.clear();
    _action.assign(actionWidth * actionHeight, 0.0f);
}

void Agent::simStep(float reward, const std::vector<float> &inputs, bool learn) {
    assert(inputs.size() == _inputWidth * _inputHeight);

    std::vector<float> inputsf = inputs;

    // Write input
    _pCs->getQueue().enqueueWriteImage(_inputImage, CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_inputWidth), static_cast<cl::size_type>(_inputHeight), 1 }, 0, 0, inputsf.data());

    _as.simStep(*_pCs, reward, _inputImage, _rng, learn);

    // Get action
    _pCs->getQueue().enqueueReadImage(_as.getAction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actionWidth), static_cast<cl::size_type>(_actionHeight), 1 }, 0, 0, _action.data());
}

std::vector<float> Agent::getStates(int li) {
    std::vector<float> states(_as.getHierarchy().getLayerDesc(li)._size.x * _as.getHierarchy().getLayerDesc(li)._size.y * 2);

    _pCs->getQueue().enqueueReadImage(_as.getHierarchy().getLayer(li)._sp.getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_as.getHierarchy().getLayerDesc(li)._size.x), static_cast<cl::size_type>(_as.getHierarchy().getLayerDesc(li)._size.y), 1 }, 0, 0, states.data());

    return states;
}

void Agent::load(const schemas::agent::Agent* fbAgent, ComputeSystem &cs) {
    _as.load(fbAgent->_as(), cs);

    _inputWidth = fbAgent->_inputWidth();
    _inputHeight = fbAgent->_inputHeight();
    _actionWidth = fbAgent->_actionWidth();
    _actionHeight = fbAgent->_actionHeight();
    _actionTileWidth = fbAgent->_actionTileWidth();
    _actionTileHeight = fbAgent->_actionTileHeight();

    ogmaneo::load(_inputImage, fbAgent->_inputImage(), cs);

    _action.clear();
    for (flatbuffers::uoffset_t i = 0; i < fbAgent->_action()->Length(); i++) {
        _action.push_back(fbAgent->_action()->Get(i));
    }
}

flatbuffers::Offset<schemas::agent::Agent> Agent::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    return schemas::agent::CreateAgent(builder,
        _as.save(builder, cs),
        _inputWidth, _inputHeight,
        _actionWidth, _actionHeight,
        _actionTileWidth, _actionTileHeight,
        ogmaneo::save(_inputImage, builder, cs),
        builder.CreateVector(_action));
}

void Agent::load(ComputeSystem &cs, ComputeProgram &prog, const std::string &fileName) {
    FILE* file = fopen(fileName.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0L, SEEK_SET);
    std::vector<uint8_t> data(length);
    fread(data.data(), sizeof(uint8_t), length, file);
    fclose(file);

    flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);

    bool verified =
        schemas::agent::VerifyAgentBuffer(verifier) |
        schemas::agent::AgentBufferHasIdentifier(data.data());

    if (verified) {
        const schemas::agent::Agent* na = schemas::agent::GetAgent(data.data());

        load(na, cs);
    }

    return; //verified;
}

void Agent::save(ComputeSystem &cs, const std::string &fileName) {
    flatbuffers::FlatBufferBuilder builder;

    flatbuffers::Offset<schemas::agent::Agent> agent = save(builder, cs);

    // Instruct the builder that this Agent is complete.
    schemas::agent::FinishAgentBuffer(builder, agent);

    // Get the built agent and size
    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier = flatbuffers::Verifier(buf, size);

    bool verified =
        schemas::agent::VerifyAgentBuffer(verifier) |
        schemas::agent::AgentBufferHasIdentifier(buf);

    if (verified) {
        FILE* file = fopen(fileName.c_str(), "wb");
        fwrite(buf, sizeof(uint8_t), size, file);
        fclose(file);
    }

    return; //verified;
}