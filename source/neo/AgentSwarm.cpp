// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "AgentSwarm.h"
#include "schemas/AgentSwarm_generated.h"
#include "schemas/AgentLayer_generated.h"

#include <iostream>

using namespace ogmaneo;

void AgentSwarm::createRandom(ComputeSystem &cs, ComputeProgram &program,
    cl_int2 inputSize, cl_int2 actionSize, cl_int2 actionTileSize, cl_int actionRadius,
    const std::vector<AgentLayerDesc> &aLayerDescs, const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
    cl_float2 initWeightRange, std::mt19937 &rng)
{
    assert(aLayerDescs.size() > 0);
    assert(aLayerDescs.size() == hLayerDescs.size());

    // Create underlying hierarchy
    _h.createRandom(cs, program, std::vector<FeatureHierarchy::InputDesc>{ FeatureHierarchy::InputDesc(inputSize, hLayerDescs.front()._inputDescs.front()._radius), FeatureHierarchy::InputDesc(actionSize, actionRadius) }, hLayerDescs, initWeightRange, rng);

    _aLayerDescs = aLayerDescs;

    _aLayers.resize(_aLayerDescs.size());

    for (int l = 0; l < _aLayers.size(); l++) {
        std::vector<AgentLayer::VisibleLayerDesc> agentVisibleLayerDescs(1);

        agentVisibleLayerDescs[0]._radius = aLayerDescs[l]._radius;
        agentVisibleLayerDescs[0]._alpha = aLayerDescs[l]._qAlpha;
        agentVisibleLayerDescs[0]._size = (l == 0) ? hLayerDescs[l]._size : cl_int2{ hLayerDescs[l]._size.x * 2, hLayerDescs[l]._size.y * 2 };

        _aLayers[l].createRandom(cs, program, (l == _aLayerDescs.size() - 1) ? actionSize : hLayerDescs[l + 1]._size, (l == _aLayerDescs.size() - 1) ? actionTileSize : cl_int2{ 2, 2 }, agentVisibleLayerDescs, initWeightRange, rng);
    }

    _ones = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), actionSize.x, actionSize.y);

    cs.getQueue().enqueueFillImage(_ones, cl_float4{ 1.0f, 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(actionSize.x), static_cast<cl::size_type>(actionSize.y), 1 });
}

void AgentSwarm::simStep(ComputeSystem &cs, float reward, const cl::Image2D &input, std::mt19937 &rng, bool learn) {
    // Activate hierarchy
    _h.simStep(cs, { input, _aLayers.back().getOneHotActions() }, rng, learn);

    // Update agent layers
    for (int l = 0; l < _aLayers.size(); l++) {
        cl::Image2D feedBack = (l == 0) ? _h.getLayer(l)._sp.getHiddenStates()[_back] : _aLayers[l - 1].getOneHotActions();

        if (l == _aLayers.size() - 1)
            _aLayers[l].simStep(cs, reward, std::vector<cl::Image2D>(1, feedBack), _ones, _aLayerDescs[l]._qGamma, _aLayerDescs[l]._qLambda, _aLayerDescs[l]._epsilon, rng, learn);
        else
            _aLayers[l].simStep(cs, reward, std::vector<cl::Image2D>(1, feedBack), _h.getLayer(l + 1)._sp.getHiddenStates()[_back], _aLayerDescs[l]._qGamma, _aLayerDescs[l]._qLambda, _aLayerDescs[l]._epsilon, rng, learn);
    }
}

void AgentSwarm::AgentLayerDesc::load(const schemas::AgentLayerDesc* fbAgentLayerDesc) {
    _radius = fbAgentLayerDesc->_radius();
    _qAlpha = fbAgentLayerDesc->_qAlpha();
    _qGamma = fbAgentLayerDesc->_qGamma();
    _qLambda = fbAgentLayerDesc->_qLambda();
    _epsilon = fbAgentLayerDesc->_epsilon();
}

void AgentSwarm::load(const schemas::AgentSwarm* fbAgentSwarm, ComputeSystem &cs) {
    if (!_aLayers.empty()) {
        assert(_aLayerDescs.size() == fbAgentSwarm->_aLayerDescs()->Length());
        assert(_aLayers.size() == fbAgentSwarm->_aLayers()->Length());
    }
    else {
        _aLayerDescs.reserve(fbAgentSwarm->_aLayerDescs()->Length());
        _aLayers.reserve(fbAgentSwarm->_aLayers()->Length());
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_aLayerDescs()->Length(); i++) {
        _aLayerDescs[i].load(fbAgentSwarm->_aLayerDescs()->Get(i));
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_aLayers()->Length(); i++) {
        _aLayers[i].load(fbAgentSwarm->_aLayers()->Get(i), cs);
    }

    _h.load(fbAgentSwarm->_h(), cs);
    ogmaneo::load(_ones, fbAgentSwarm->_ones(), cs);
}

schemas::AgentLayerDesc AgentSwarm::AgentLayerDesc::save(flatbuffers::FlatBufferBuilder& builder) {
    schemas::AgentLayerDesc layerDesc(_radius, _qAlpha, _qGamma, _qLambda, _epsilon);
    return layerDesc;
}

flatbuffers::Offset<schemas::AgentSwarm> AgentSwarm::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    std::vector<schemas::AgentLayerDesc> aLayerDescs;

    for (AgentLayerDesc layerDesc : _aLayerDescs)
        aLayerDescs.push_back(layerDesc.save(builder));

    std::vector<flatbuffers::Offset<schemas::agent::AgentLayer>> aLayers;

    for (AgentLayer layer : _aLayers)
        aLayers.push_back(layer.save(builder, cs));

    return schemas::CreateAgentSwarm(builder,
        _h.save(builder, cs),
        ogmaneo::save(_ones, builder, cs),
        builder.CreateVector(aLayers),
        builder.CreateVectorOfStructs(aLayerDescs));
}