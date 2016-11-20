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

void AgentSwarm::createRandom(ComputeSystem &cs, ComputeProgram &hProgram, ComputeProgram &pProgram, ComputeProgram &asProgram,
    const std::vector<cl_int2> &actionSizes, const std::vector<cl_int2> actionTileSizes,
    const std::vector<std::vector<AgentLayerDesc>> &aLayerDescs,
    const std::vector<Predictor::PredLayerDesc> &pLayerDescs,
    const std::vector<FeatureHierarchy::LayerDesc> &hLayerDescs,
    cl_float2 initWeightRange, std::mt19937 &rng)
{
    assert(aLayerDescs.size() > 0);
    assert(aLayerDescs.size() == hLayerDescs.size());

    // Create underlying hierarchy
    _p.createRandom(cs, hProgram, pProgram, pLayerDescs, hLayerDescs, initWeightRange, rng);

    _aLayerDescs = aLayerDescs;

    _aLayers.resize(_aLayerDescs.size());

    for (int l = 0; l < _aLayers.size(); l++) {
        _aLayers[l].resize(_aLayerDescs[l].size());

        for (int i = 0; i < _aLayers[l].size(); i++) {
            std::vector<AgentLayer::VisibleLayerDesc> agentVisibleLayerDescs(1);

            float lrScalar = l == 0 ? 0.25f : 1.0f;

            agentVisibleLayerDescs[0]._radius = aLayerDescs[l][i]._radius;
            agentVisibleLayerDescs[0]._qAlpha = aLayerDescs[l][i]._qAlpha * lrScalar;
            agentVisibleLayerDescs[0]._actionAlpha = aLayerDescs[l][i]._actionAlpha * lrScalar;

            cl_int2 size = _p.getHierarchy().getLayer(l)._sf->getHiddenSize();

            agentVisibleLayerDescs[0]._size = (l == 0) ? size : cl_int2{ size.x * 2, size.y * 2 };

            _aLayers[l][i].createRandom(cs, asProgram, (l == _aLayers.size() - 1) ? actionSizes[i] : _p.getHierarchy().getLayer(l + 1)._sf->getHiddenSize(), (l == _aLayers.size() - 1) ? actionTileSizes[i] : cl_int2{ 2, 2 }, agentVisibleLayerDescs, initWeightRange, rng);
        }
    }

    _ones.resize(_aLayers.back().size());

    for (int i = 0; i < _ones.size(); i++) {
        _ones[i] = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), actionSizes[i].x, actionSizes[i].y);

        cs.getQueue().enqueueFillImage(_ones[i], cl_float4{ 1.0f, 1.0f, 1.0f, 1.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(actionSizes[i].x), static_cast<cl::size_type>(actionSizes[i].y), 1 });
    }

    _rewardSums.clear();
    _rewardSums.assign(_aLayers.size(), 0.0f);

    _rewardCounts.clear();
    _rewardCounts.assign(_aLayers.size(), 0.0f);
}

void AgentSwarm::simStep(ComputeSystem &cs, float reward, const std::vector<cl::Image2D> &inputs, const std::vector<cl::Image2D> &inputsCorrupted, std::mt19937 &rng, bool learn) {
    // Activate hierarchy
    _p.simStep(cs, inputs, inputsCorrupted, rng, learn);

    // Update agent layers
    for (int l = 0; l < _aLayers.size(); l++) {
        _rewardSums[l] += reward;
        _rewardCounts[l] += 1.0f;

        //if (_p.getHierarchy().getLayer(l)._tpReset || _p.getHierarchy().getLayer(l)._tpNextReset || (l < _aLayers.size() - 1 && (_p.getHierarchy().getLayer(l + 1)._tpReset || _p.getHierarchy().getLayer(l + 1)._tpNextReset))) {
            float totalReward = _rewardSums[l] / _rewardCounts[l];

            for (int i = 0; i < _aLayers[l].size(); i++) {
                cl::Image2D inputs = (l == 0) ? _p.getHierarchy().getLayer(l)._sf->getHiddenStates()[_back] : _aLayers[l - 1].front().getOneHotActions();

                if (l == _aLayers.size() - 1)
                    _aLayers[l][i].simStep(cs, totalReward, std::vector<cl::Image2D>(1, inputs), _ones[i], _aLayerDescs[l][i]._qGamma, _aLayerDescs[l][i]._qLambda, _aLayerDescs[l][i]._actionLambda, _aLayerDescs[l][i]._maxActionWeightMag, rng, learn);
                else
                    _aLayers[l][i].simStep(cs, totalReward, std::vector<cl::Image2D>(1, inputs), _p.getHierarchy().getLayer(l + 1)._sf->getHiddenStates()[_back], _aLayerDescs[l][i]._qGamma, _aLayerDescs[l][i]._qLambda, _aLayerDescs[l][i]._actionLambda, _aLayerDescs[l][i]._maxActionWeightMag, rng, learn);
            }

            _rewardSums[l] = 0.0f;
            _rewardCounts[l] = 0.0f;
        //}
    }
}

void AgentSwarm::AgentLayerDesc::load(const schemas::AgentSwarmLayerDesc* fbAgentSwarmLayerDesc) {
    _radius = fbAgentSwarmLayerDesc->_radius();
    _qAlpha = fbAgentSwarmLayerDesc->_qAlpha();
    _qGamma = fbAgentSwarmLayerDesc->_qGamma();
    _qLambda = fbAgentSwarmLayerDesc->_qLambda();
    _actionLambda = fbAgentSwarmLayerDesc->_actionLambda();
    _maxActionWeightMag = fbAgentSwarmLayerDesc->_maxActionWeightMag();
}

flatbuffers::Offset<schemas::AgentSwarmLayerDesc> AgentSwarm::AgentLayerDesc::save(flatbuffers::FlatBufferBuilder &builder) {
    return schemas::CreateAgentSwarmLayerDesc(builder, _radius, _qAlpha, _qGamma, _qLambda, _actionLambda, _maxActionWeightMag);
}

void AgentSwarm::load(const schemas::AgentSwarm* fbAgentSwarm, ComputeSystem &cs) {
    if (!_aLayers.empty()) {
        assert(_aLayerDescs.size() == fbAgentSwarm->_aLayerDescs()->Length());
        assert(_aLayers.size() == fbAgentSwarm->_aLayers()->Length());
        assert(_rewardSums.size() == fbAgentSwarm->_rewardSums()->Length());
        assert(_rewardCounts.size() == fbAgentSwarm->_rewardCounts()->Length());
        assert(_ones.size() == fbAgentSwarm->_ones()->Length());
    }
    else {
        _aLayerDescs.reserve(fbAgentSwarm->_aLayerDescs()->Length());
        _aLayers.reserve(fbAgentSwarm->_aLayers()->Length());
        _rewardSums.reserve(fbAgentSwarm->_rewardSums()->Length());
        _rewardCounts.reserve(fbAgentSwarm->_rewardCounts()->Length());
        _ones.reserve(fbAgentSwarm->_ones()->Length());
    }

    _p.load(fbAgentSwarm->_p(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_aLayerDescs()->Length(); i++) {
        const schemas::AgentSwarmLayerDescs* fbAgentSwarmLayerDescs = fbAgentSwarm->_aLayerDescs()->Get(i);
        assert(_aLayerDescs[i].size() == fbAgentSwarmLayerDescs->_layerDescs()->Length());
        _aLayerDescs[i].reserve(fbAgentSwarmLayerDescs->_layerDescs()->Length());

        for (flatbuffers::uoffset_t j = 0; j < fbAgentSwarmLayerDescs->_layerDescs()->Length(); j++) {
            _aLayerDescs[i][j].load(fbAgentSwarmLayerDescs->_layerDescs()->Get(j));
        }
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_aLayers()->Length(); i++) {
        const schemas::AgentSwarmLayers* fbAgentSwarmLayers = fbAgentSwarm->_aLayers()->Get(i);
        assert(_aLayers[i].size() == fbAgentSwarmLayers->_layers()->Length());
        _aLayers[i].reserve(fbAgentSwarmLayers->_layers()->Length());

        for (flatbuffers::uoffset_t j = 0; j < fbAgentSwarmLayers->_layers()->Length(); j++) {
            _aLayers[i][j].load(fbAgentSwarmLayers->_layers()->Get(j), cs);
        }
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_rewardSums()->Length(); i++) {
        _rewardSums[i] = fbAgentSwarm->_rewardSums()->Get(i);
    }
    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_rewardCounts()->Length(); i++) {
        _rewardCounts[i] = fbAgentSwarm->_rewardCounts()->Get(i);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentSwarm->_ones()->Length(); i++) {
        ogmaneo::load(_ones[i], fbAgentSwarm->_ones()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::AgentSwarm> AgentSwarm::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::AgentSwarmLayerDescs>> aLayerDescs;
    for (std::vector<AgentLayerDesc> layerDescs : _aLayerDescs)
    {
        std::vector<flatbuffers::Offset<schemas::AgentSwarmLayerDesc>> aLayerDesc;
        for (AgentLayerDesc layerDesc : layerDescs)
            aLayerDesc.push_back(layerDesc.save(builder));

        aLayerDescs.push_back(schemas::CreateAgentSwarmLayerDescs(builder, builder.CreateVector(aLayerDesc)));
    }

    std::vector<flatbuffers::Offset<schemas::AgentSwarmLayers>> aLayers;
    for (std::vector<AgentLayer> layers : _aLayers)
    {
        std::vector<flatbuffers::Offset<schemas::AgentLayer>> aLayer;
        for (AgentLayer layer : layers)
            aLayer.push_back(layer.save(builder, cs));

        aLayers.push_back(schemas::CreateAgentSwarmLayers(builder, builder.CreateVector(aLayer)));
    }

    flatbuffers::Offset<flatbuffers::Vector<float>> rewardSums = builder.CreateVector(_rewardSums.data(), _rewardSums.size());
    flatbuffers::Offset<flatbuffers::Vector<float>> rewardCounts = builder.CreateVector(_rewardCounts.data(), _rewardCounts.size());

    std::vector<flatbuffers::Offset<schemas::Image2D>> ones;
    for (cl::Image2D image : _ones)
        ones.push_back(ogmaneo::save(image, builder, cs));

    return schemas::CreateAgentSwarm(builder,
        _p.save(builder, cs),
        builder.CreateVector(aLayers),
        builder.CreateVector(aLayerDescs),
        rewardSums, rewardCounts,
        builder.CreateVector(ones));
}