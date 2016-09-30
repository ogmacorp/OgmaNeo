// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PredictorLayer.h"

using namespace ogmaneo;

void PredictorLayer::createRandom(ComputeSystem &cs, ComputeProgram &program,
    cl_int2 hiddenSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs,
    cl_float2 initWeightRange, std::mt19937 &rng)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

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

            vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

            randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }
    }

    // Hidden state data
    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    // Create kernels
    _stimulusKernel = cl::Kernel(program.getProgram(), "plStimulus");
    _learnPredWeightsKernel = cl::Kernel(program.getProgram(), "plLearnPredWeights");
    _thresholdKernel = cl::Kernel(program.getProgram(), "plThreshold");
}

void PredictorLayer::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, bool threshold) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

    // Start by clearing stimulus summation buffer to biases
    cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _stimulusKernel.setArg(argIndex++, visibleStates[vli]);
            _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
            _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
            _stimulusKernel.setArg(argIndex++, vl._weights[_back]);
            _stimulusKernel.setArg(argIndex++, vld._size);
            _stimulusKernel.setArg(argIndex++, vl._hiddenToVisible);
            _stimulusKernel.setArg(argIndex++, vld._radius);

            cs.getQueue().enqueueNDRangeKernel(_stimulusKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
    }

    if (threshold) {
        int argIndex = 0;

        _thresholdKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _thresholdKernel.setArg(argIndex++, _hiddenStates[_front]);

        cs.getQueue().enqueueNDRangeKernel(_thresholdKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
    else {
        // Copy to hidden states
        cs.getQueue().enqueueCopyImage(_hiddenSummationTemp[_back], _hiddenStates[_front], zeroOrigin, zeroOrigin, hiddenRegion);
    }
}

void PredictorLayer::stepEnd(ComputeSystem &cs) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

    std::swap(_hiddenStates[_front], _hiddenStates[_back]);

    // Swap buffers
    /*for (int vli = 0; vli < _visibleLayers.size(); vli++) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];
    }*/
}

void PredictorLayer::learn(ComputeSystem &cs, const cl::Image2D &targets, const std::vector<cl::Image2D> &visibleStatesPrev) {
    // Learn weights
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _learnPredWeightsKernel.setArg(argIndex++, visibleStatesPrev[vli]);
        _learnPredWeightsKernel.setArg(argIndex++, targets);
        _learnPredWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
        _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_back]);
        _learnPredWeightsKernel.setArg(argIndex++, vl._weights[_front]);
        _learnPredWeightsKernel.setArg(argIndex++, vld._size);
        _learnPredWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
        _learnPredWeightsKernel.setArg(argIndex++, vld._radius);
        _learnPredWeightsKernel.setArg(argIndex++, vld._alpha);

        cs.getQueue().enqueueNDRangeKernel(_learnPredWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(vl._weights[_front], vl._weights[_back]);
    }
}

void PredictorLayer::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    /*for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];
    }*/
}

void PredictorLayer::VisibleLayerDesc::load(const schemas::predictor::VisibleLayerDesc* fbVisibleLayerDesc, ComputeSystem &cs) {
    _size.x = fbVisibleLayerDesc->_size().x();
    _size.y = fbVisibleLayerDesc->_size().y();
    _radius = fbVisibleLayerDesc->_radius();
    _alpha = fbVisibleLayerDesc->_alpha();
}

void PredictorLayer::VisibleLayer::load(const schemas::predictor::VisibleLayer* fbVisibleLayer, ComputeSystem &cs) {
    ogmaneo::load(_weights, fbVisibleLayer->_weights(), cs);

    _hiddenToVisible.x = fbVisibleLayer->_hiddenToVisible()->x();
    _hiddenToVisible.y = fbVisibleLayer->_hiddenToVisible()->y();
    _visibleToHidden.x = fbVisibleLayer->_visibleToHidden()->x();
    _visibleToHidden.y = fbVisibleLayer->_visibleToHidden()->y();
    _reverseRadii.x = fbVisibleLayer->_reverseRadii()->x();
    _reverseRadii.y = fbVisibleLayer->_reverseRadii()->y();
}

void PredictorLayer::load(const schemas::predictor::Layer* fbPL, ComputeSystem &cs) {
    if (!_visibleLayers.empty()) {
        assert(_hiddenSize.x == fbPL->_hiddenSize()->x());
        assert(_hiddenSize.y == fbPL->_hiddenSize()->y());
        assert(_visibleLayerDescs.size() == fbPL->_visibleLayerDescs()->Length());
        assert(_visibleLayers.size() == fbPL->_visibleLayers()->Length());
    }
    else {
        _hiddenSize.x = fbPL->_hiddenSize()->x();
        _hiddenSize.y = fbPL->_hiddenSize()->y();
        _visibleLayerDescs.reserve(fbPL->_visibleLayerDescs()->Length());
        _visibleLayers.reserve(fbPL->_visibleLayers()->Length());
    }

    ogmaneo::load(_hiddenSummationTemp, fbPL->_hiddenSummationTemp(), cs);
    ogmaneo::load(_hiddenStates, fbPL->_hiddenStates(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbPL->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbPL->_visibleLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPL->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbPL->_visibleLayers()->Get(i), cs);
    }
}

schemas::predictor::VisibleLayerDesc PredictorLayer::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::int2 size(_size.x, _size.y);
    schemas::predictor::VisibleLayerDesc visibleLayerDesc(size, _radius, _alpha);
    return visibleLayerDesc;
}

flatbuffers::Offset<schemas::predictor::VisibleLayer> PredictorLayer::VisibleLayer::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::predictor::CreateVisibleLayer(builder,
        ogmaneo::save(_weights, builder, cs),
        &hiddenToVisible,
        &visibleToHidden,
        &reverseRadii
    );
}

flatbuffers::Offset<schemas::predictor::Layer> PredictorLayer::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);

    std::vector<schemas::predictor::VisibleLayerDesc> visibleLayerDescs;

    for (PredictorLayer::VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::predictor::VisibleLayer>> visibleLayers;

    for (PredictorLayer::VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    return schemas::predictor::CreateLayer(builder,
        &hiddenSize,
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        ogmaneo::save(_hiddenStates, builder, cs),
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers)
    );
}
