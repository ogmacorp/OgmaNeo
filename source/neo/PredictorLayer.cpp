// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PredictorLayer.h"
#include "SparseFeaturesChunk.h"

using namespace ogmaneo;

void PredictorLayer::createRandom(ComputeSystem &cs, ComputeProgram &plProgram,
    cl_int2 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    Type type, cl_int2 chunkSize,
    cl_float2 initWeightRange, std::mt19937 &rng)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _type = type;

    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _chunkSize = chunkSize;

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    _visibleLayers.resize(_visibleLayerDescs.size());

    cl::Kernel randomUniform2DKernel = cl::Kernel(plProgram.getProgram(), "randomUniform2D");
    cl::Kernel randomUniform3DKernel = cl::Kernel(plProgram.getProgram(), "randomUniform3D");

    int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / static_cast<float>(_chunkSize.x)));
    int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / static_cast<float>(_chunkSize.y)));

    // Create layers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        vl._hiddenToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(chunksInX),
            static_cast<float>(vld._size.y) / static_cast<float>(chunksInY)
        };

        vl._visibleToHidden = cl_float2{ static_cast<float>(chunksInX) / static_cast<float>(vld._size.x),
            static_cast<float>(chunksInY) / static_cast<float>(vld._size.y)
        };

        vl._reverseRadii = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
        };

        {
            int weightDiam = vld._radius * 2 + 1;

            int numWeights = weightDiam * weightDiam;

            cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

            vl._weights = createDoubleBuffer3D(cs, weightsSize, _type == _q ? CL_RG : CL_R, CL_FLOAT);

            randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, { initWeightRange.x, 0.0f, 0.0f, 0.0f }, { initWeightRange.y, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, rng);
        }

        vl._derivedInput = createDoubleBuffer2D(cs, vld._size, CL_RG, CL_FLOAT);

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
    }

    // Hidden state data
    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, _type == _inhibitBinary ? CL_RG : CL_R, CL_FLOAT);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
 
    // Create kernels
    _deriveInputsKernel = cl::Kernel(plProgram.getProgram(), "plDeriveInputs");
    _stimulusKernel = cl::Kernel(plProgram.getProgram(), "plStimulus");

    if (_type == _inhibitBinary) {
        _learnPredWeightsKernel = cl::Kernel(plProgram.getProgram(), "plLearnPredWeightsBinary");
        _inhibitBinaryKernel = cl::Kernel(plProgram.getProgram(), "plInhibitBinary");
    }
    else if (_type == _q)
        _learnPredWeightsKernel = cl::Kernel(plProgram.getProgram(), "plLearnPredWeightsQ");
   else
        _learnPredWeightsKernel = cl::Kernel(plProgram.getProgram(), "plLearnPredWeights");

    _propagateKernel = cl::Kernel(plProgram.getProgram(), "plPropagate");
}

void PredictorLayer::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, std::mt19937 &rng) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Start by clearing stimulus summation buffer
    cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Derive inputs
        {
            int argIndex = 0;

            _deriveInputsKernel.setArg(argIndex++, visibleStates[vli]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_back]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _deriveInputsKernel.setArg(argIndex++, vld._gamma);

            cs.getQueue().enqueueNDRangeKernel(_deriveInputsKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }

        {
            int argIndex = 0;

            _stimulusKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
            _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
            _stimulusKernel.setArg(argIndex++, vl._weights[_back]);
            _stimulusKernel.setArg(argIndex++, vld._size);
            _stimulusKernel.setArg(argIndex++, vl._hiddenToVisible);
            _stimulusKernel.setArg(argIndex++, vld._radius);
            _stimulusKernel.setArg(argIndex++, _chunkSize);

            cs.getQueue().enqueueNDRangeKernel(_stimulusKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

            // Swap buffers
            std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
        }  
    }

    if (_type == _inhibitBinary) {
        // Inhibit
        int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / static_cast<float>(_chunkSize.x)));
        int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / static_cast<float>(_chunkSize.y)));

        int argIndex = 0;

        _inhibitBinaryKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _inhibitBinaryKernel.setArg(argIndex++, _hiddenStates[_front]);
        _inhibitBinaryKernel.setArg(argIndex++, _hiddenSize);
        _inhibitBinaryKernel.setArg(argIndex++, _chunkSize);

        cs.getQueue().enqueueNDRangeKernel(_inhibitBinaryKernel, cl::NullRange, cl::NDRange(chunksInX, chunksInY));
    }
    else
        cs.getQueue().enqueueCopyImage(_hiddenSummationTemp[_back], _hiddenStates[_front], { 0, 0, 0 }, { 0, 0, 0 }, { static_cast<cl::size_type>(_hiddenSize.x), static_cast<cl::size_type>(_hiddenSize.y), 1 });
}

void PredictorLayer::propagate(ComputeSystem &cs, const cl::Image2D &hiddenStates, const cl::Image2D &hiddenTargets, int vli, DoubleBuffer2D &visibleStates, std::mt19937 &rng) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    int argIndex = 0;

    _propagateKernel.setArg(argIndex++, hiddenStates);
    _propagateKernel.setArg(argIndex++, hiddenTargets);
    _propagateKernel.setArg(argIndex++, visibleStates[_back]);
    _propagateKernel.setArg(argIndex++, visibleStates[_front]);
    _propagateKernel.setArg(argIndex++, vl._weights[_back]);
    _propagateKernel.setArg(argIndex++, vld._size);
    _propagateKernel.setArg(argIndex++, _hiddenSize);
    _propagateKernel.setArg(argIndex++, vl._visibleToHidden);
    _propagateKernel.setArg(argIndex++, vl._hiddenToVisible);
    _propagateKernel.setArg(argIndex++, vld._radius);
    _propagateKernel.setArg(argIndex++, vl._reverseRadii);

    cs.getQueue().enqueueNDRangeKernel(_propagateKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
}

void PredictorLayer::stepEnd(ComputeSystem &cs) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    std::swap(_hiddenStates[_front], _hiddenStates[_back]);

    // Swap buffers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
    }
}

void PredictorLayer::learn(ComputeSystem &cs, const cl::Image2D &targets, bool predictFromPrevious, float tdError) {
    if (_type == _inhibitBinary) {
        // Learn weights
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnPredWeightsKernel.setArg(argIndex++, predictFromPrevious ? vl._derivedInput[_front] : vl._derivedInput[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, targets);
            _learnPredWeightsKernel.setArg(argIndex++, predictFromPrevious ? _hiddenStates[_front] : _hiddenStates[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_front]);
            _learnPredWeightsKernel.setArg(argIndex++, vld._size);
            _learnPredWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnPredWeightsKernel.setArg(argIndex++, vld._radius);
            _learnPredWeightsKernel.setArg(argIndex++, _chunkSize);
            _learnPredWeightsKernel.setArg(argIndex++, vld._alpha);

            cs.getQueue().enqueueNDRangeKernel(_learnPredWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

            std::swap(vl._weights[_front], vl._weights[_back]);
        }
    }
    else if (_type == _q) {
        // Learn weights
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnPredWeightsKernel.setArg(argIndex++, vl._derivedInput[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, targets); // Describes selected action (tiled one hot) for Q
            //_learnPredWeightsQKernel.setArg(argIndex++, _hiddenStates[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_front]);
            _learnPredWeightsKernel.setArg(argIndex++, vld._size);
            _learnPredWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnPredWeightsKernel.setArg(argIndex++, vld._radius);
            _learnPredWeightsKernel.setArg(argIndex++, _chunkSize);
            _learnPredWeightsKernel.setArg(argIndex++, vld._alpha);
            _learnPredWeightsKernel.setArg(argIndex++, tdError);
            _learnPredWeightsKernel.setArg(argIndex++, vld._lambda);

            cs.getQueue().enqueueNDRangeKernel(_learnPredWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

            std::swap(vl._weights[_front], vl._weights[_back]);
        }
    }
    else {
        // Learn weights
        for (int vli = 0; vli < _visibleLayers.size(); vli++) {
            VisibleLayer &vl = _visibleLayers[vli];
            VisibleLayerDesc &vld = _visibleLayerDescs[vli];

            int argIndex = 0;

            _learnPredWeightsKernel.setArg(argIndex++, predictFromPrevious ? vl._derivedInput[_front] : vl._derivedInput[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, targets);
            _learnPredWeightsKernel.setArg(argIndex++, predictFromPrevious ? _hiddenStates[_front] : _hiddenStates[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_back]);
            _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_front]);
            _learnPredWeightsKernel.setArg(argIndex++, vld._size);
            _learnPredWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnPredWeightsKernel.setArg(argIndex++, vld._radius);
            _learnPredWeightsKernel.setArg(argIndex++, _chunkSize);
            _learnPredWeightsKernel.setArg(argIndex++, vld._alpha);

            cs.getQueue().enqueueNDRangeKernel(_learnPredWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

            std::swap(vl._weights[_front], vl._weights[_back]);
        }
    }
}

void PredictorLayer::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
}

void PredictorLayer::VisibleLayerDesc::load(const schemas::VisiblePredictorLayerDesc* fbVisiblePredictorLayerDesc, ComputeSystem &cs) {
    _size = cl_int2{ fbVisiblePredictorLayerDesc->_size().x(), fbVisiblePredictorLayerDesc->_size().y() };
    _radius = fbVisiblePredictorLayerDesc->_radius();
    _alpha = fbVisiblePredictorLayerDesc->_alpha();
    _lambda = fbVisiblePredictorLayerDesc->_lambda();
    _gamma = fbVisiblePredictorLayerDesc->_gamma();
}

schemas::VisiblePredictorLayerDesc PredictorLayer::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 size(_size.x, _size.y);
    return schemas::VisiblePredictorLayerDesc(size, _radius, _alpha, _lambda, _gamma);
}

void PredictorLayer::VisibleLayer::load(const schemas::VisiblePredictorLayer* fbVisiblePredictorLayer, ComputeSystem &cs) {
    _hiddenToVisible = cl_float2{ fbVisiblePredictorLayer->_hiddenToVisible()->x(), fbVisiblePredictorLayer->_hiddenToVisible()->y() };
    _visibleToHidden = cl_float2{ fbVisiblePredictorLayer->_visibleToHidden()->x(), fbVisiblePredictorLayer->_visibleToHidden()->y() };
    _reverseRadii = cl_int2{ fbVisiblePredictorLayer->_reverseRadii()->x(), fbVisiblePredictorLayer->_reverseRadii()->y() };
    ogmaneo::load(_derivedInput, fbVisiblePredictorLayer->_derivedInput(), cs);
    ogmaneo::load(_weights, fbVisiblePredictorLayer->_weights(), cs);
}

flatbuffers::Offset<schemas::VisiblePredictorLayer> PredictorLayer::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::CreateVisiblePredictorLayer(builder,
        ogmaneo::save(_derivedInput, builder, cs),
        ogmaneo::save(_weights, builder, cs),
        &hiddenToVisible, &visibleToHidden, &reverseRadii);
}

void PredictorLayer::load(const schemas::PredictorLayer* fbPredictorLayer, ComputeSystem &cs) {
    assert(_hiddenSize.x == fbPredictorLayer->_hiddenSize()->x());
    assert(_hiddenSize.y == fbPredictorLayer->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbPredictorLayer->_visibleLayerDescs()->Length());
    assert(_visibleLayers.size() == fbPredictorLayer->_visibleLayers()->Length());

    switch (fbPredictorLayer->_type()) {
    default:
    case schemas::PredictorLayerType::PredictorLayerType__none: _type = Type::_none; break;
    case schemas::PredictorLayerType::PredictorLayerType__inhibitBinary: _type = Type::_inhibitBinary; break;
    case schemas::PredictorLayerType::PredictorLayerType__q: _type = Type::_q; break;
    }

    _hiddenSize = cl_int2{ fbPredictorLayer->_hiddenSize()->x(), fbPredictorLayer->_hiddenSize()->y() };

    ogmaneo::load(_hiddenSummationTemp, fbPredictorLayer->_hiddenSummationTemp(), cs);
    ogmaneo::load(_hiddenStates, fbPredictorLayer->_hiddenStates(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbPredictorLayer->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbPredictorLayer->_visibleLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPredictorLayer->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbPredictorLayer->_visibleLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::PredictorLayer> PredictorLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);
    schemas::PredictorLayerType type;

    switch (_type) {
    default:
    case _none: type = schemas::PredictorLayerType::PredictorLayerType__none; break;
    case _inhibitBinary: type = schemas::PredictorLayerType::PredictorLayerType__inhibitBinary; break;
    case _q: type = schemas::PredictorLayerType::PredictorLayerType__q; break;
    }


    std::vector<schemas::VisiblePredictorLayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::VisiblePredictorLayer>> visibleLayers;
    for (VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    return schemas::CreatePredictorLayer(builder,
        type, &hiddenSize,
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        ogmaneo::save(_hiddenStates, builder, cs),
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers));
}