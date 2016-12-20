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

            vl._qWeights = createDoubleBuffer3D(cs, weightsSize, CL_RG, CL_FLOAT);

            randomUniform(vl._qWeights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }

        vl._derivedInput = createDoubleBuffer2D(cs, vld._size, CL_RG, CL_FLOAT);

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
    }

    // Hidden state data
    _qStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);
    _actionTaken = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);
    _actionTakenMax = createDoubleBuffer2D(cs, _numActionTiles, CL_R, CL_FLOAT);
    _spreadStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _oneHotAction = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _tdError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _numActionTiles.x, _numActionTiles.y);

    cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_actionTaken[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_actionTakenMax[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_spreadStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    _hiddenSummationTempQ = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    // Create kernels
    _deriveInputsKernel = cl::Kernel(program.getProgram(), "alDeriveInputs");
    _activateKernel = cl::Kernel(program.getProgram(), "alActivate");
    _learnQKernel = cl::Kernel(program.getProgram(), "alLearnQ");
    _actionToOneHotKernel = cl::Kernel(program.getProgram(), "alActionToOneHot");
    _getActionKernel = cl::Kernel(program.getProgram(), "alGetAction");
    _setActionKernel = cl::Kernel(program.getProgram(), "alSetAction");
    _spreadKernel = cl::Kernel(program.getProgram(), "alSpread");
}

void AgentLayer::simStep(ComputeSystem &cs, float reward, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &modulator,
    float qGamma, float qLambda, float epsilon, float chunkGamma, cl_int2 chunkSize, std::mt19937 &rng, bool learn)
{
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };
    cl::array<cl::size_type, 3> actionRegion = { static_cast<cl_uint>(_numActionTiles.x), static_cast<cl_uint>(_numActionTiles.y), 1 };

    cs.getQueue().enqueueFillImage(_hiddenSummationTempQ[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find Q
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Derive inputs
        {
            int argIndex = 0;

            _deriveInputsKernel.setArg(argIndex++, visibleStates[vli]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_back]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_front]);

            cs.getQueue().enqueueNDRangeKernel(_deriveInputsKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }

        {
            int argIndex = 0;

            _activateKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _activateKernel.setArg(argIndex++, vl._qWeights[_back]);
            _activateKernel.setArg(argIndex++, _hiddenSummationTempQ[_back]);
            _activateKernel.setArg(argIndex++, _hiddenSummationTempQ[_front]);
            _activateKernel.setArg(argIndex++, vld._size);
            _activateKernel.setArg(argIndex++, vl._hiddenToVisible);
            _activateKernel.setArg(argIndex++, vld._radius);

            cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTempQ[_front], _hiddenSummationTempQ[_back]);
    }

    // Copy to Q states
    cs.getQueue().enqueueCopyImage(_hiddenSummationTempQ[_back], _qStates[_front], zeroOrigin, zeroOrigin, hiddenRegion);

    // Get newest actions
    {
        std::uniform_int_distribution<int> seedDist(0, 9999);

        cl_uint2 seed = { static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng)) };

        int argIndex = 0;

        _getActionKernel.setArg(argIndex++, _qStates[_front]);
        _getActionKernel.setArg(argIndex++, _actionTaken[_front]);
        _getActionKernel.setArg(argIndex++, _actionTakenMax[_front]);
        _getActionKernel.setArg(argIndex++, _actionTileSize);
        _getActionKernel.setArg(argIndex++, epsilon);
        _getActionKernel.setArg(argIndex++, seed);

        cs.getQueue().enqueueNDRangeKernel(_getActionKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));

        std::swap(_actionTaken[_front], _actionTaken[_back]);
        std::swap(_actionTakenMax[_front], _actionTakenMax[_back]);
    }

    // Compute TD errors
    {
        int argIndex = 0;

        _setActionKernel.setArg(argIndex++, modulator);
        _setActionKernel.setArg(argIndex++, _actionTaken[_back]);
        _setActionKernel.setArg(argIndex++, _actionTaken[_front]);
        _setActionKernel.setArg(argIndex++, _actionTakenMax[_back]);
        _setActionKernel.setArg(argIndex++, _actionTakenMax[_front]);
        _setActionKernel.setArg(argIndex++, _qStates[_front]);
        _setActionKernel.setArg(argIndex++, _qStates[_back]);
        _setActionKernel.setArg(argIndex++, _tdError);
        _setActionKernel.setArg(argIndex++, _oneHotAction[_front]);
        _setActionKernel.setArg(argIndex++, _actionTileSize);
        _setActionKernel.setArg(argIndex++, chunkSize);
        _setActionKernel.setArg(argIndex++, reward);
        _setActionKernel.setArg(argIndex++, qGamma);

        cs.getQueue().enqueueNDRangeKernel(_setActionKernel, cl::NullRange, cl::NDRange(_numActionTiles.x, _numActionTiles.y));

        std::swap(_oneHotAction[_front], _oneHotAction[_back]);
    }

    std::swap(_qStates[_front], _qStates[_back]);

    // Spread actions (unsparsify)
    {
        int argIndex = 0;

        _spreadKernel.setArg(argIndex++, _oneHotAction[_back]);
        _spreadKernel.setArg(argIndex++, _spreadStates[_front]);
        _spreadKernel.setArg(argIndex++, _numActionTiles);
        _spreadKernel.setArg(argIndex++, _actionTileSize);
        _spreadKernel.setArg(argIndex++, chunkGamma);
        _spreadKernel.setArg(argIndex++, chunkSize);

        cs.getQueue().enqueueNDRangeKernel(_spreadKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(_spreadStates[_front], _spreadStates[_back]);
    }

    if (learn) {
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            // Learn Q
            {
                int argIndex = 0;

                _learnQKernel.setArg(argIndex++, vl._derivedInput[_front]);
                _learnQKernel.setArg(argIndex++, _spreadStates[_back]);
                _learnQKernel.setArg(argIndex++, _tdError);
                _learnQKernel.setArg(argIndex++, vl._qWeights[_back]);
                _learnQKernel.setArg(argIndex++, vl._qWeights[_front]);
                _learnQKernel.setArg(argIndex++, vld._size);
                _learnQKernel.setArg(argIndex++, vl._hiddenToVisible);
                _learnQKernel.setArg(argIndex++, vld._radius);
                _learnQKernel.setArg(argIndex++, vld._qAlpha);
                _learnQKernel.setArg(argIndex++, qLambda);
                _learnQKernel.setArg(argIndex++, _actionTileSize);

                cs.getQueue().enqueueNDRangeKernel(_learnQKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
            }

            std::swap(vl._qWeights[_front], vl._qWeights[_back]);

            std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
        }
    }
}

void AgentLayer::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };
    cl::array<cl::size_type, 3> actionRegion = { static_cast<cl_uint>(_numActionTiles.x), static_cast<cl_uint>(_numActionTiles.y), 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_qStates[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_actionTaken[_back], zeroColor, zeroOrigin, actionRegion);
    cs.getQueue().enqueueFillImage(_actionTakenMax[_back], zeroColor, zeroOrigin, actionRegion);
}

void AgentLayer::VisibleLayerDesc::load(const schemas::VisibleAgentLayerDesc* fbVisibleAgentLayerDesc) {
    _size = cl_int2{ fbVisibleAgentLayerDesc->_size().x(), fbVisibleAgentLayerDesc->_size().y() };
    _radius = fbVisibleAgentLayerDesc->_radius();
    _qAlpha = fbVisibleAgentLayerDesc->_qAlpha();
}

schemas::VisibleAgentLayerDesc AgentLayer::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder) {
    schemas::int2 size(_size.x, _size.y);
    return schemas::VisibleAgentLayerDesc(size, _radius, _qAlpha);
}

void AgentLayer::VisibleLayer::load(const schemas::VisibleAgentLayer* fbVisibleAgentLayer, ComputeSystem &cs) {
    _hiddenToVisible = cl_float2{ fbVisibleAgentLayer->_hiddenToVisible()->x(), fbVisibleAgentLayer->_hiddenToVisible()->y() };
    _visibleToHidden = cl_float2{ fbVisibleAgentLayer->_visibleToHidden()->x(), fbVisibleAgentLayer->_visibleToHidden()->y() };
    _reverseRadii = cl_int2{ fbVisibleAgentLayer->_reverseRadii()->x(), fbVisibleAgentLayer->_reverseRadii()->y() };
    ogmaneo::load(_qWeights, fbVisibleAgentLayer->_qWeights(), cs);
}

flatbuffers::Offset<schemas::VisibleAgentLayer> AgentLayer::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::CreateVisibleAgentLayer(builder,
        ogmaneo::save(_qWeights, builder, cs),
        &hiddenToVisible, &visibleToHidden, &reverseRadii);
}

void AgentLayer::load(const schemas::AgentLayer* fbAgentLayer, ComputeSystem &cs) {
    assert(_visibleLayerDescs.size() == fbAgentLayer->_visibleLayerDescs()->Length());
    assert(_visibleLayers.size() == fbAgentLayer->_visibleLayers()->Length());

    _numActionTiles = cl_int2{ fbAgentLayer->_numActionTiles()->x(), fbAgentLayer->_numActionTiles()->y() };
    _actionTileSize = cl_int2{ fbAgentLayer->_actionTileSize()->x(), fbAgentLayer->_actionTileSize()->y() };
    _hiddenSize = cl_int2{ fbAgentLayer->_hiddenSize()->x(), fbAgentLayer->_hiddenSize()->y() };
    ogmaneo::load(_qStates, fbAgentLayer->_qStates(), cs);
    ogmaneo::load(_actionTaken, fbAgentLayer->_actionTaken(), cs);
    ogmaneo::load(_actionTakenMax, fbAgentLayer->_actionTakenMax(), cs);
    ogmaneo::load(_oneHotAction, fbAgentLayer->_oneHotAction(), cs);
    ogmaneo::load(_tdError, fbAgentLayer->_tdError(), cs);
    ogmaneo::load(_hiddenSummationTempQ, fbAgentLayer->_hiddenSummationTempQ(), cs);

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
        ogmaneo::save(_actionTaken, builder, cs),
        ogmaneo::save(_actionTakenMax, builder, cs),
        ogmaneo::save(_oneHotAction, builder, cs),
        ogmaneo::save(_tdError, builder, cs),
        ogmaneo::save(_hiddenSummationTempQ, builder, cs),
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers));
}