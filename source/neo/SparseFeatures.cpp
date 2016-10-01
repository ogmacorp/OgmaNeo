// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseFeatures.h"

using namespace ogmaneo;

void SparseFeatures::createRandom(ComputeSystem &cs, ComputeProgram &program,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
    cl_int inhibitionRadius,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _inhibitionRadius = inhibitionRadius;

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

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

            randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, { 0.0f, 1.0f }, rng);
        }

        vl._derivedInput = createDoubleBuffer2D(cs, vld._size, CL_R, CL_FLOAT);

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
    }

    // Hidden state data
    _hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    randomUniform(_hiddenBiases[_back], cs, randomUniform2DKernel, _hiddenSize, initWeightRange, rng);

    // Create kernels
    _stimulusKernel = cl::Kernel(program.getProgram(), "spStimulus");
    _activateKernel = cl::Kernel(program.getProgram(), "spActivate");
    _inhibitKernel = cl::Kernel(program.getProgram(), "spInhibit");
    _learnWeightsKernel = cl::Kernel(program.getProgram(), "spLearnWeights");
    _learnBiasesKernel = cl::Kernel(program.getProgram(), "spLearnBiases");
    _deriveInputsKernel = cl::Kernel(program.getProgram(), "spDeriveInputs");
}

void SparseFeatures::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float activeRatio, std::mt19937 &rng) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Start by clearing stimulus summation buffer to biases
    cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _deriveInputsKernel.setArg(argIndex++, visibleStates[vli]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_back]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_front]);

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
            _stimulusKernel.setArg(argIndex++, vld._ignoreMiddle);

            cs.getQueue().enqueueNDRangeKernel(_stimulusKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
    }

    // Activate
    {
        int argIndex = 0;

        _activateKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _activateKernel.setArg(argIndex++, _hiddenStates[_back]);
        _activateKernel.setArg(argIndex++, _hiddenBiases[_back]);
        _activateKernel.setArg(argIndex++, _hiddenActivations[_back]);
        _activateKernel.setArg(argIndex++, _hiddenActivations[_front]);

        cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Inhibit
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);
        _inhibitKernel.setArg(argIndex++, _inhibitionRadius);
        _inhibitKernel.setArg(argIndex++, activeRatio);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void SparseFeatures::stepEnd(ComputeSystem &cs) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
    std::swap(_hiddenStates[_front], _hiddenStates[_back]);

    // Swap buffers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
    }
}

void SparseFeatures::learn(ComputeSystem &cs, float biasAlpha, float activeRatio)
{
    // Learn weights
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _learnWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
            _learnWeightsKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
            _learnWeightsKernel.setArg(argIndex++, vld._size);
            _learnWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnWeightsKernel.setArg(argIndex++, vld._radius);
            _learnWeightsKernel.setArg(argIndex++, activeRatio);
            _learnWeightsKernel.setArg(argIndex++, vld._weightAlpha);

            cs.getQueue().enqueueNDRangeKernel(_learnWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        std::swap(vl._weights[_front], vl._weights[_back]);
    }

    // Bias update
    {
        int argIndex = 0;

        _learnBiasesKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _learnBiasesKernel.setArg(argIndex++, _hiddenStates[_front]);
        _learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
        _learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
        _learnBiasesKernel.setArg(argIndex++, activeRatio);
        _learnBiasesKernel.setArg(argIndex++, biasAlpha);

        cs.getQueue().enqueueNDRangeKernel(_learnBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
    }
}

void SparseFeatures::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
    }
}

void SparseFeatures::VisibleLayerDesc::load(const schemas::hierarchy::VisibleLayerDesc* fbVisibleLayerDesc, ComputeSystem &cs) {
    _size.x = fbVisibleLayerDesc->_size().x();
    _size.y = fbVisibleLayerDesc->_size().y();
    _radius = fbVisibleLayerDesc->_radius();
    _ignoreMiddle = fbVisibleLayerDesc->_ignoreMiddle();
    _weightAlpha = fbVisibleLayerDesc->_weightAlpha();
}

void SparseFeatures::VisibleLayer::load(const schemas::hierarchy::VisibleLayer* fbVisibleLayer, ComputeSystem &cs) {
    ogmaneo::load(_derivedInput, fbVisibleLayer->_derivedInput(), cs);

    ogmaneo::load(_weights, fbVisibleLayer->_weights(), cs);

    _hiddenToVisible.x = fbVisibleLayer->_hiddenToVisible()->x();
    _hiddenToVisible.y = fbVisibleLayer->_hiddenToVisible()->y();
    _visibleToHidden.x = fbVisibleLayer->_visibleToHidden()->x();
    _visibleToHidden.y = fbVisibleLayer->_visibleToHidden()->y();
    _reverseRadii.x = fbVisibleLayer->_reverseRadii()->x();
    _reverseRadii.y = fbVisibleLayer->_reverseRadii()->y();
}

void SparseFeatures::load(const schemas::hierarchy::SparseFeatures* fbSF, ComputeSystem &cs) {
    if (!_visibleLayers.empty()) {
        assert(_hiddenSize.x == fbSF->_hiddenSize()->x());
        assert(_hiddenSize.y == fbSF->_hiddenSize()->y());
        assert(_visibleLayerDescs.size() == fbSF->_visibleLayerDescs()->Length());
        assert(_visibleLayers.size() == fbSF->_visibleLayers()->Length());
    }
    else {
        _hiddenSize.x = fbSF->_hiddenSize()->x();
        _hiddenSize.y = fbSF->_hiddenSize()->y();
        _visibleLayerDescs.reserve(fbSF->_visibleLayerDescs()->Length());
        _visibleLayers.reserve(fbSF->_visibleLayers()->Length());
    }

    ogmaneo::load(_hiddenActivations, fbSF->_hiddenActivations(), cs);
    ogmaneo::load(_hiddenStates, fbSF->_hiddenStates(), cs);
    ogmaneo::load(_hiddenBiases, fbSF->_hiddenBiases(), cs);

    _inhibitionRadius = fbSF->_inhibitionRadius();

    ogmaneo::load(_hiddenSummationTemp, fbSF->_hiddenSummationTemp(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbSF->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSF->_visibleLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSF->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbSF->_visibleLayers()->Get(i), cs);
    }
}

schemas::hierarchy::VisibleLayerDesc SparseFeatures::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::int2 size(_size.x, _size.y);
    schemas::hierarchy::VisibleLayerDesc visibleLayerDesc(
        size, _radius, _ignoreMiddle, _weightAlpha
    );
    return visibleLayerDesc;
}

flatbuffers::Offset<schemas::hierarchy::VisibleLayer> SparseFeatures::VisibleLayer::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::hierarchy::CreateVisibleLayer(builder,
        ogmaneo::save(_derivedInput, builder, cs),
        ogmaneo::save(_weights, builder, cs),
        &hiddenToVisible,
        &visibleToHidden,
        &reverseRadii
    );
}

flatbuffers::Offset<schemas::hierarchy::SparseFeatures> SparseFeatures::save(flatbuffers::FlatBufferBuilder& builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);

    std::vector<schemas::hierarchy::VisibleLayerDesc> visibleLayerDescs;

    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::hierarchy::VisibleLayer>> visibleLayers;

    for (VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    return schemas::hierarchy::CreateSparseFeatures(
        builder,
        ogmaneo::save(_hiddenActivations, builder, cs),
        ogmaneo::save(_hiddenStates, builder, cs),
        ogmaneo::save(_hiddenBiases, builder, cs),
        &hiddenSize,
        _inhibitionRadius,
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers)
    );
}
