// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "AgentLayer.h"

using namespace ogmaneo;

void AgentLayer::createRandom(ComputeSystem &cs, ComputeProgram &program,
    cl_int2 numActionTiles, cl_int2 actionTileSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _visibleLayerDescs = visibleLayerDescs;

    _numActionTiles = numActionTiles;
    _actionTileSize = actionTileSize;

    _hiddenSize = { _numActionTiles.x * _actionTileSize.x, _numActionTiles.y * _actionTileSize.y };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };
    cl::array<cl::size_type, 3> actionRegion = { _numActionTiles.x, _numActionTiles.y, 1 };

    _visibleLayers.resize(_visibleLayerDescs.size());

    cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
    cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
        };

        vl._visibleToHidden = cl_float2{ static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y)
        };

        vl._reverseRadii = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
        };

        {
            int weightDiam = vld._radius * 2 + 1;

            int numWeights = weightDiam * weightDiam;

            cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

            vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

            randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }
    }

    // Hidden state data
    _qStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
    _action = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);
    _actionTaken = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);

    _tdError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _hiddenSize.x, _hiddenSize.y);
    _oneHotAction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _hiddenSize.x, _hiddenSize.y);

    cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_action[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_actionTaken[_back], zeroColor, zeroOrigin, actionRegion);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    // Create kernels
    _findQKernel = cl::Kernel(program.getProgram(), "alFindQ");
    _learnQKernel = cl::Kernel(program.getProgram(), "alLearnQ");
    _actionToOneHotKernel = cl::Kernel(program.getProgram(), "alActionToOneHot");
    _getActionKernel = cl::Kernel(program.getProgram(), "alGetAction");
    _setActionKernel = cl::Kernel(program.getProgram(), "alSetAction");
    _actionExplorationKernel = cl::Kernel(program.getProgram(), "alActionExploration");
}

void AgentLayer::simStep(ComputeSystem &cs, float reward, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &modulator,
    float qGamma, float qLambda, float epsilon, std::mt19937 &rng, bool learn)
{
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

    cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find Q
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _findQKernel.setArg(argIndex++, visibleStates[vli]);
            _findQKernel.setArg(argIndex++, vl._weights[_back]);
            _findQKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
            _findQKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
            _findQKernel.setArg(argIndex++, vld._size);
            _findQKernel.setArg(argIndex++, vl._hiddenToVisible);
            _findQKernel.setArg(argIndex++, vld._radius);

            cs.getQueue().enqueueNDRangeKernel(_findQKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
    }

    // Copy to hidden states
    cs.getQueue().enqueueCopyImage(_hiddenSummationTemp[_back], _qStates[_front], zeroOrigin, zeroOrigin, hiddenRegion);

    // Get newest actions
    {
        int argIndex = 0;

        _getActionKernel.setArg(argIndex++, _qStates[_front]);
        _getActionKernel.setArg(argIndex++, _action[_front]);
        _getActionKernel.setArg(argIndex++, _actionTileSize);

        cs.getQueue().enqueueNDRangeKernel(_getActionKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));

        std::swap(_action[_front], _action[_back]);
    }

    // Exploration
    {
        std::uniform_int_distribution<int> seedDist(0, 9999);

        cl_uint2 seed = { seedDist(rng), seedDist(rng) };

        int argIndex = 0;

        _actionExplorationKernel.setArg(argIndex++, _action[_back]);
        _actionExplorationKernel.setArg(argIndex++, _actionTaken[_front]);
        _actionExplorationKernel.setArg(argIndex++, epsilon);
        _actionExplorationKernel.setArg(argIndex++, _actionTileSize.x * _actionTileSize.y);
        _actionExplorationKernel.setArg(argIndex++, seed);

        cs.getQueue().enqueueNDRangeKernel(_actionExplorationKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));

        std::swap(_actionTaken[_front], _actionTaken[_back]);
    }

    // Compute TD errors
    {
        int argIndex = 0;

        _setActionKernel.setArg(argIndex++, modulator);
        _setActionKernel.setArg(argIndex++, _action[_back]);
        _setActionKernel.setArg(argIndex++, _action[_front]);
        _setActionKernel.setArg(argIndex++, _actionTaken[_back]);
        _setActionKernel.setArg(argIndex++, _actionTaken[_front]);
        _setActionKernel.setArg(argIndex++, _qStates[_front]);
        _setActionKernel.setArg(argIndex++, _qStates[_back]);
        _setActionKernel.setArg(argIndex++, _tdError);
        _setActionKernel.setArg(argIndex++, _oneHotAction);
        _setActionKernel.setArg(argIndex++, _actionTileSize);
        _setActionKernel.setArg(argIndex++, reward);
        _setActionKernel.setArg(argIndex++, qGamma);

        cs.getQueue().enqueueNDRangeKernel(_setActionKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));
    }

    std::swap(_qStates[_front], _qStates[_back]);

    if (learn) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            // Learn Q
            {
                int argIndex = 0;

                _learnQKernel.setArg(argIndex++, visibleStates[vli]);
                _learnQKernel.setArg(argIndex++, _qStates[_back]);
                _learnQKernel.setArg(argIndex++, _qStates[_front]);
                _learnQKernel.setArg(argIndex++, _tdError);
                _learnQKernel.setArg(argIndex++, _oneHotAction);
                _learnQKernel.setArg(argIndex++, vl._weights[_back]);
                _learnQKernel.setArg(argIndex++, vl._weights[_front]);
                _learnQKernel.setArg(argIndex++, vld._size);
                _learnQKernel.setArg(argIndex++, vl._hiddenToVisible);
                _learnQKernel.setArg(argIndex++, vld._radius);
                _learnQKernel.setArg(argIndex++, vld._alpha);
                _learnQKernel.setArg(argIndex++, qLambda);

                cs.getQueue().enqueueNDRangeKernel(_learnQKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
            }

            std::swap(vl._weights[_front], vl._weights[_back]);
        }
    }
}

void AgentLayer::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };
    cl::array<cl::size_type, 3> actionRegion = { _numActionTiles.x, _numActionTiles.y, 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_action[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_actionTaken[_back], zeroColor, zeroOrigin, actionRegion);

    /*for (int vli = 0; vli < _visibleLayers.size(); vli++) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];
    }*/
}

void AgentLayer::VisibleLayerDesc::load(const schemas::agent::VisibleLayerDesc* fbVisibleLayerDesc) {
    _size.x = fbVisibleLayerDesc->_size().x();
    _size.y = fbVisibleLayerDesc->_size().y();
    _radius = fbVisibleLayerDesc->_radius();
    _alpha = fbVisibleLayerDesc->_alpha();
}

schemas::agent::VisibleLayerDesc AgentLayer::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder) {
    schemas::int2 size(_size.x, _size.y);
    schemas::agent::VisibleLayerDesc visibleLayerDesc(size, _radius, _alpha);
    return visibleLayerDesc;
}

void AgentLayer::VisibleLayer::load(const schemas::agent::VisibleLayer* fbVisibleLayer, ComputeSystem &cs) {
    ogmaneo::load(_weights, fbVisibleLayer->_weights(), cs);
    _hiddenToVisible.x = fbVisibleLayer->_hiddenToVisible()->x();
    _hiddenToVisible.y = fbVisibleLayer->_hiddenToVisible()->y();
    _visibleToHidden.x = fbVisibleLayer->_visibleToHidden()->x();
    _visibleToHidden.y = fbVisibleLayer->_visibleToHidden()->y();
    _reverseRadii.x = fbVisibleLayer->_reverseRadii()->x();
    _reverseRadii.y = fbVisibleLayer->_reverseRadii()->y();
}

flatbuffers::Offset<schemas::agent::VisibleLayer> AgentLayer::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::agent::CreateVisibleLayer(builder,
        ogmaneo::save(_weights, builder, cs),
        &hiddenToVisible,
        &visibleToHidden,
        &reverseRadii);
}

void AgentLayer::load(const schemas::agent::AgentLayer* fbAgentLayer, ComputeSystem &cs) {
    if (!_visibleLayers.empty()) {
        assert(_visibleLayerDescs.size() == fbAgentLayer->_visibleLayerDescs()->Length());
        assert(_visibleLayers.size() == fbAgentLayer->_visibleLayers()->Length());
    }
    else {
        _visibleLayerDescs.reserve(fbAgentLayer->_visibleLayerDescs()->Length());
        _visibleLayers.reserve(fbAgentLayer->_visibleLayers()->Length());
    }
        
    _numActionTiles.x = fbAgentLayer->_numActionTiles()->x();
    _numActionTiles.y = fbAgentLayer->_numActionTiles()->y();
    _actionTileSize.x = fbAgentLayer->_actionTileSize()->x();
    _actionTileSize.y = fbAgentLayer->_actionTileSize()->y();
    _hiddenSize.x = fbAgentLayer->_hiddenSize()->x();
    _hiddenSize.y = fbAgentLayer->_hiddenSize()->y();
    
    ogmaneo::load(_qStates, fbAgentLayer->_qStates(), cs);
    ogmaneo::load(_action, fbAgentLayer->_action(), cs);
    ogmaneo::load(_actionTaken, fbAgentLayer->_actionTaken(), cs);
    ogmaneo::load(_tdError, fbAgentLayer->_tdError(), cs);
    ogmaneo::load(_oneHotAction, fbAgentLayer->_oneHotAction(), cs);
    ogmaneo::load(_hiddenSummationTemp, fbAgentLayer->_hiddenSummationTemp(), cs);
    
    for (flatbuffers::uoffset_t i = 0; i < fbAgentLayer->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbAgentLayer->_visibleLayers()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentLayer->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbAgentLayer->_visibleLayerDescs()->Get(i));
    }
}

flatbuffers::Offset<schemas::agent::AgentLayer> AgentLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 numActionTiles(_numActionTiles.x, _numActionTiles.y);
    schemas::int2 actionTileSize(_actionTileSize.x, _actionTileSize.y);
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);

    std::vector<schemas::agent::VisibleLayerDesc> layerDescs;

    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        layerDescs.push_back(layerDesc.save(builder));

    std::vector<flatbuffers::Offset<schemas::agent::VisibleLayer>> layers;

    for (VisibleLayer layer : _visibleLayers)
        layers.push_back(layer.save(builder, cs));

    // Build the schemas::AgentLayer
    flatbuffers::Offset<schemas::agent::AgentLayer> fbAgentLayer = schemas::agent::CreateAgentLayer(builder,
        &numActionTiles, &actionTileSize, &hiddenSize,
        ogmaneo::save(_qStates,builder, cs),
        ogmaneo::save(_action, builder, cs),
        ogmaneo::save(_actionTaken, builder, cs),
        ogmaneo::save(_tdError, builder, cs),
        ogmaneo::save(_oneHotAction, builder, cs),
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        builder.CreateVector(layers),
        builder.CreateVectorOfStructs(layerDescs));

    return fbAgentLayer;
}