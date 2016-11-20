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
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };
    cl::array<cl::size_type, 3> actionRegion = { static_cast<cl_uint>(_numActionTiles.x), static_cast<cl_uint>(_numActionTiles.y), 1 };

    _visibleLayers.resize(_visibleLayerDescs.size());

    cl::Kernel randomUniform2DKernel = cl::Kernel(program.getProgram(), "randomUniform2D");
    cl::Kernel randomUniform3DKernel = cl::Kernel(program.getProgram(), "randomUniform3D");

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._qToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(_numActionTiles.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_numActionTiles.y)
        };

        vl._visibleToQ = cl_float2{ static_cast<float>(_numActionTiles.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_numActionTiles.y) / static_cast<float>(vld._size.y)
        };

        vl._reverseRadiiQ = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToQ.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToQ.y * vld._radius) + 1)
        };

        vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(_hiddenSize.x),
            static_cast<float>(vld._size.y) / static_cast<float>(_hiddenSize.y)
        };

        vl._visibleToHidden = cl_float2{ static_cast<float>(_hiddenSize.x) / static_cast<float>(vld._size.x),
            static_cast<float>(_hiddenSize.y) / static_cast<float>(vld._size.y)
        };

        vl._reverseRadiiHidden = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
        };

        {
            int weightDiam = vld._radius * 2 + 1;

            int numWeights = weightDiam * weightDiam;

            cl_int3 weightsSize = { _numActionTiles.x, _numActionTiles.y, numWeights };

            vl._qWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

            randomUniform(vl._qWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }

        {
            int weightDiam = vld._radius * 2 + 1;

            int numWeights = weightDiam * weightDiam;

            cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

            vl._actionWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

            randomUniform(vl._actionWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }
    }

    // Hidden state data
    _qStates = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);
    _actionProbabilities = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
    _actionTaken = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);

    _tdError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _numActionTiles.x, _numActionTiles.y);
    _oneHotAction = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), _hiddenSize.x, _hiddenSize.y);

    cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_actionProbabilities[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_actionTaken[_back], zeroColor, zeroOrigin, actionRegion);

    _hiddenSummationTempQ = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);
    _hiddenSummationTempHidden = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    // Create kernels
    _activateKernel = cl::Kernel(program.getProgram(), "alActivate");
    _learnQKernel = cl::Kernel(program.getProgram(), "alLearnQ");
    _learnActionsKernel = cl::Kernel(program.getProgram(), "alLearnActions");
    _actionToOneHotKernel = cl::Kernel(program.getProgram(), "alActionToOneHot");
    _getActionKernel = cl::Kernel(program.getProgram(), "alGetAction");
    _setActionKernel = cl::Kernel(program.getProgram(), "alSetAction");
}

void AgentLayer::simStep(ComputeSystem &cs, float reward, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &modulator,
    float qGamma, float qLambda, float actionLambda, float maxActionWeightMag, std::mt19937 &rng, bool learn)
{
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };
    cl::array<cl::size_type, 3> actionRegion = { static_cast<cl_uint>(_numActionTiles.x), static_cast<cl_uint>(_numActionTiles.y), 1 };

    cs.getQueue().enqueueFillImage(_hiddenSummationTempQ[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_hiddenSummationTempHidden[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find Q
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _activateKernel.setArg(argIndex++, visibleStates[vli]);
            _activateKernel.setArg(argIndex++, vl._qWeights[_back]);
            _activateKernel.setArg(argIndex++, _hiddenSummationTempQ[_back]);
            _activateKernel.setArg(argIndex++, _hiddenSummationTempQ[_front]);
            _activateKernel.setArg(argIndex++, vld._size);
            _activateKernel.setArg(argIndex++, vl._qToVisible);
            _activateKernel.setArg(argIndex++, vld._radius);

            cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTempQ[_front], _hiddenSummationTempQ[_back]);

        {
            int argIndex = 0;

            _activateKernel.setArg(argIndex++, visibleStates[vli]);
            _activateKernel.setArg(argIndex++, vl._actionWeights[_back]);
            _activateKernel.setArg(argIndex++, _hiddenSummationTempHidden[_back]);
            _activateKernel.setArg(argIndex++, _hiddenSummationTempHidden[_front]);
            _activateKernel.setArg(argIndex++, vld._size);
            _activateKernel.setArg(argIndex++, vl._hiddenToVisible);
            _activateKernel.setArg(argIndex++, vld._radius);

            cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTempHidden[_front], _hiddenSummationTempHidden[_back]);
    }

    // Copy to Q states
    cs.getQueue().enqueueCopyImage(_hiddenSummationTempQ[_back], _qStates[_front], zeroOrigin, zeroOrigin, actionRegion);

    // Get newest actions
    {
        std::uniform_int_distribution<int> seedDist(0, 9999);

        cl_uint2 seed = { static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng)) };

        int argIndex = 0;

        _getActionKernel.setArg(argIndex++, _hiddenSummationTempHidden[_back]);
        _getActionKernel.setArg(argIndex++, _actionProbabilities[_front]);
        _getActionKernel.setArg(argIndex++, _actionTaken[_front]);
        _getActionKernel.setArg(argIndex++, _actionTileSize);
        _getActionKernel.setArg(argIndex++, seed);

        cs.getQueue().enqueueNDRangeKernel(_getActionKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));

        std::swap(_actionTaken[_front], _actionTaken[_back]);
    }

    // Compute TD errors
    {
        int argIndex = 0;

        _setActionKernel.setArg(argIndex++, modulator);
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
    std::swap(_actionProbabilities[_front], _actionProbabilities[_back]);

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
                _learnQKernel.setArg(argIndex++, vl._qWeights[_back]);
                _learnQKernel.setArg(argIndex++, vl._qWeights[_front]);
                _learnQKernel.setArg(argIndex++, vld._size);
                _learnQKernel.setArg(argIndex++, vl._qToVisible);
                _learnQKernel.setArg(argIndex++, vld._radius);
                _learnQKernel.setArg(argIndex++, vld._qAlpha);
                _learnQKernel.setArg(argIndex++, qLambda);

                cs.getQueue().enqueueNDRangeKernel(_learnQKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));
            }

            std::swap(vl._qWeights[_front], vl._qWeights[_back]);

            // Learn action
            {
                int argIndex = 0;

                _learnActionsKernel.setArg(argIndex++, visibleStates[vli]);
                _learnActionsKernel.setArg(argIndex++, _actionProbabilities[_front]);
                _learnActionsKernel.setArg(argIndex++, _tdError);
                _learnActionsKernel.setArg(argIndex++, _oneHotAction);
                _learnActionsKernel.setArg(argIndex++, vl._actionWeights[_back]);
                _learnActionsKernel.setArg(argIndex++, vl._actionWeights[_front]);
                _learnActionsKernel.setArg(argIndex++, vld._size);
                _learnActionsKernel.setArg(argIndex++, vl._hiddenToVisible);
                _learnActionsKernel.setArg(argIndex++, vld._radius);
                _learnActionsKernel.setArg(argIndex++, vld._actionAlpha);
                _learnActionsKernel.setArg(argIndex++, actionLambda);
                _learnActionsKernel.setArg(argIndex++, _actionTileSize);
                _learnActionsKernel.setArg(argIndex++, maxActionWeightMag);

                cs.getQueue().enqueueNDRangeKernel(_learnActionsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
            }

            std::swap(vl._actionWeights[_front], vl._actionWeights[_back]);
        }
    }
}

void AgentLayer::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };
    cl::array<cl::size_type, 3> actionRegion = { static_cast<cl_uint>(_numActionTiles.x), static_cast<cl_uint>(_numActionTiles.y), 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_actionProbabilities[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_actionTaken[_back], zeroColor, zeroOrigin, actionRegion);
}

void AgentLayer::VisibleLayerDesc::load(const schemas::VisibleAgentLayerDesc* fbVisibleAgentLayerDesc) {
    _size = cl_int2{ fbVisibleAgentLayerDesc->_size().x(), fbVisibleAgentLayerDesc->_size().y() };
    _radius = fbVisibleAgentLayerDesc->_radius();
    _qAlpha = fbVisibleAgentLayerDesc->_qAlpha();
    _actionAlpha = fbVisibleAgentLayerDesc->_actionAlpha();
}

schemas::VisibleAgentLayerDesc AgentLayer::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder) {
    schemas::int2 size(_size.x, _size.y);
    return schemas::VisibleAgentLayerDesc(size, _radius, _qAlpha, _actionAlpha);
}

void AgentLayer::VisibleLayer::load(const schemas::VisibleAgentLayer* fbVisibleAgentLayer, ComputeSystem &cs) {
    _qToVisible = cl_float2{ fbVisibleAgentLayer->_qToVisible()->x(), fbVisibleAgentLayer->_qToVisible()->y() };
    _visibleToQ = cl_float2{ fbVisibleAgentLayer->_visibleToQ()->x(), fbVisibleAgentLayer->_visibleToQ()->y() };
    _reverseRadiiQ = cl_int2{ fbVisibleAgentLayer->_reverseRadiiQ()->x(), fbVisibleAgentLayer->_reverseRadiiQ()->y() };
    _hiddenToVisible = cl_float2{ fbVisibleAgentLayer->_hiddenToVisible()->x(), fbVisibleAgentLayer->_hiddenToVisible()->y() };
    _visibleToHidden = cl_float2{ fbVisibleAgentLayer->_visibleToHidden()->x(), fbVisibleAgentLayer->_visibleToHidden()->y() };
    _reverseRadiiHidden = cl_int2{ fbVisibleAgentLayer->_reverseRadiiHidden()->x(), fbVisibleAgentLayer->_reverseRadiiHidden()->y() };
    ogmaneo::load(_qWeights, fbVisibleAgentLayer->_qWeights(), cs);
    ogmaneo::load(_actionWeights, fbVisibleAgentLayer->_actionWeights(), cs);
}

flatbuffers::Offset<schemas::VisibleAgentLayer> AgentLayer::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 qToVisible(_qToVisible.x, _qToVisible.y);
    schemas::float2 visibleToQ(_visibleToQ.x, _visibleToQ.y);
    schemas::int2 reverseRadiiQ(_reverseRadiiQ.x, _reverseRadiiQ.y);
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadiiHidden(_reverseRadiiHidden.x, _reverseRadiiHidden.y);

    return schemas::CreateVisibleAgentLayer(builder,
        ogmaneo::save(_qWeights, builder, cs),
        ogmaneo::save(_actionWeights, builder, cs),
        &qToVisible, &visibleToQ, &reverseRadiiQ,
        &hiddenToVisible, &visibleToHidden, &reverseRadiiHidden);
}

void AgentLayer::load(const schemas::AgentLayer* fbAgentLayer, ComputeSystem &cs) {
    if (!_visibleLayers.empty()) {
        assert(_visibleLayerDescs.size() == fbAgentLayer->_visibleLayerDescs()->Length());
        assert(_visibleLayers.size() == fbAgentLayer->_visibleLayers()->Length());
    }
    else {
        _visibleLayerDescs.reserve(fbAgentLayer->_visibleLayerDescs()->Length());
        _visibleLayers.reserve(fbAgentLayer->_visibleLayers()->Length());
    }

    _numActionTiles = cl_int2{ fbAgentLayer->_numActionTiles()->x(), fbAgentLayer->_numActionTiles()->y() };
    _actionTileSize = cl_int2{ fbAgentLayer->_actionTileSize()->x(), fbAgentLayer->_actionTileSize()->y() };
    _hiddenSize = cl_int2{ fbAgentLayer->_hiddenSize()->x(), fbAgentLayer->_hiddenSize()->y() };
    ogmaneo::load(_qStates, fbAgentLayer->_qStates(), cs);
    ogmaneo::load(_actionProbabilities, fbAgentLayer->_actionProbabilities(), cs);
    ogmaneo::load(_actionTaken, fbAgentLayer->_actionTaken(), cs);
    ogmaneo::load(_tdError, fbAgentLayer->_tdError(), cs);
    ogmaneo::load(_oneHotAction, fbAgentLayer->_oneHotAction(), cs);
    ogmaneo::load(_hiddenSummationTempQ, fbAgentLayer->_hiddenSummationTempQ(), cs);
    ogmaneo::load(_hiddenSummationTempHidden, fbAgentLayer->_hiddenSummationTempHidden(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbAgentLayer->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbAgentLayer->_visibleLayerDescs()->Get(i));
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgentLayer->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbAgentLayer->_visibleLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::AgentLayer> AgentLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 numActionTiles(_numActionTiles.x, _numActionTiles.y);
    schemas::int2 actionTileSize(_actionTileSize.x, _actionTileSize.y);
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);

    std::vector<schemas::VisibleAgentLayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder));

    std::vector<flatbuffers::Offset<schemas::VisibleAgentLayer>> visibleLayers;
    for (VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    return schemas::CreateAgentLayer(builder,
        &numActionTiles, &actionTileSize, &hiddenSize,
        ogmaneo::save(_qStates, builder, cs),
        ogmaneo::save(_actionProbabilities, builder, cs),
        ogmaneo::save(_actionTaken, builder, cs),
        ogmaneo::save(_tdError, builder, cs),
        ogmaneo::save(_oneHotAction, builder, cs),
        ogmaneo::save(_hiddenSummationTempQ, builder, cs),
        ogmaneo::save(_hiddenSummationTempHidden, builder, cs),
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers));
}