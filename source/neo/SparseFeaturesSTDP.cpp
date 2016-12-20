// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseFeaturesSTDP.h"

using namespace ogmaneo;

SparseFeaturesSTDP::SparseFeaturesSTDP(ComputeSystem &cs, ComputeProgram &sfhProgram,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
    cl_int inhibitionRadius, cl_float biasAlpha,
    cl_float activeRatio, cl_float gamma, cl_float2 initWeightRange,
    std::mt19937 &rng)
    : _hiddenSize(hiddenSize), _inhibitionRadius(inhibitionRadius), _activeRatio(activeRatio), _biasAlpha(biasAlpha), _gamma(gamma)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _type = SparseFeaturesType::_stdp;

    _visibleLayerDescs = visibleLayerDescs;

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    _visibleLayers.resize(_visibleLayerDescs.size());

    cl::Kernel randomUniform2DKernel = cl::Kernel(sfhProgram.getProgram(), "randomUniform2D");
    cl::Kernel randomUniform3DKernel = cl::Kernel(sfhProgram.getProgram(), "randomUniform3D");

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

        vl._derivedInput = createDoubleBuffer2D(cs, vld._size, CL_RG, CL_FLOAT);

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
    }

    // Hidden state data
    _hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);

    _hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    //randomUniform(_hiddenBiases[_back], cs, randomUniform2DKernel, _hiddenSize, initWeightRange, rng);
    cs.getQueue().enqueueFillImage(_hiddenBiases[_back], zeroColor, zeroOrigin, hiddenRegion);

    // Create kernels
    _stimulusKernel = cl::Kernel(sfhProgram.getProgram(), "sfsStimulus");
    _activateKernel = cl::Kernel(sfhProgram.getProgram(), "sfsActivate");
    _inhibitKernel = cl::Kernel(sfhProgram.getProgram(), "sfsInhibit");
    _inhibitOtherKernel = cl::Kernel(sfhProgram.getProgram(), "sfsInhibitOther");
    _learnWeightsKernel = cl::Kernel(sfhProgram.getProgram(), "sfsLearnWeights");
    _learnBiasesKernel = cl::Kernel(sfhProgram.getProgram(), "sfsLearnBiases");
    _deriveInputsKernel = cl::Kernel(sfhProgram.getProgram(), "sfsDeriveInputs");
}

void SparseFeaturesSTDP::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &predictionsPrev, std::mt19937 &rng) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Start by clearing stimulus summation buffer
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
            _deriveInputsKernel.setArg(argIndex++, vld._lambda);

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
        std::uniform_int_distribution<int> seedDist(0, 9999);

        cl_uint2 seed = { static_cast<cl_uint>(seedDist(rng)), static_cast<cl_uint>(seedDist(rng)) };

        int argIndex = 0;

        _activateKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _activateKernel.setArg(argIndex++, _hiddenStates[_back]);
        _activateKernel.setArg(argIndex++, _hiddenBiases[_back]);
        _activateKernel.setArg(argIndex++, _hiddenActivations[_back]);
        _activateKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _activateKernel.setArg(argIndex++, seed);

        cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Inhibit
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_back]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);
        _inhibitKernel.setArg(argIndex++, _inhibitionRadius);
        _inhibitKernel.setArg(argIndex++, _activeRatio);
        _inhibitKernel.setArg(argIndex++, _gamma);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void SparseFeaturesSTDP::stepEnd(ComputeSystem &cs) {
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

void SparseFeaturesSTDP::learn(ComputeSystem &cs, const cl::Image2D &predictionsPrev, std::mt19937 &rng) {
    // Learn weights
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Weight update
        {
            int argIndex = 0;

            _learnWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
            _learnWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _learnWeightsKernel.setArg(argIndex++, vl._derivedInput[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
            _learnWeightsKernel.setArg(argIndex++, vld._size);
            _learnWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnWeightsKernel.setArg(argIndex++, vld._radius);
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
        _learnBiasesKernel.setArg(argIndex++, _activeRatio);
        _learnBiasesKernel.setArg(argIndex++, _biasAlpha);

        cs.getQueue().enqueueNDRangeKernel(_learnBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
    }
}

void SparseFeaturesSTDP::inhibit(ComputeSystem &cs, const cl::Image2D &activations, cl::Image2D &states, std::mt19937 &rng) {
    // Inhibit
    {
        int argIndex = 0;

        _inhibitOtherKernel.setArg(argIndex++, activations);
        _inhibitOtherKernel.setArg(argIndex++, states);
        _inhibitOtherKernel.setArg(argIndex++, _hiddenSize);
        _inhibitOtherKernel.setArg(argIndex++, _inhibitionRadius);
        _inhibitOtherKernel.setArg(argIndex++, _activeRatio);

        cs.getQueue().enqueueNDRangeKernel(_inhibitOtherKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void SparseFeaturesSTDP::clearMemory(ComputeSystem &cs) {
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

void SparseFeaturesSTDP::VisibleLayerDesc::load(const schemas::VisibleSTDPLayerDesc* fbVisibleSTDPLayerDesc, ComputeSystem &cs) {
    _size = cl_int2{ fbVisibleSTDPLayerDesc->_size().x(), fbVisibleSTDPLayerDesc->_size().y() };
    _radius = fbVisibleSTDPLayerDesc->_radius();
    _ignoreMiddle = fbVisibleSTDPLayerDesc->_ignoreMiddle();
    _weightAlpha = fbVisibleSTDPLayerDesc->_weightAlpha();
}

schemas::VisibleSTDPLayerDesc SparseFeaturesSTDP::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 size(_size.x, _size.y);
    return schemas::VisibleSTDPLayerDesc(size, _radius, _ignoreMiddle, _weightAlpha);
}

void SparseFeaturesSTDP::VisibleLayer::load(const schemas::VisibleSTDPLayer* fbVisibleSTDPLayer, ComputeSystem &cs) {
    ogmaneo::load(_derivedInput, fbVisibleSTDPLayer->_derivedInput(), cs);
    ogmaneo::load(_weights, fbVisibleSTDPLayer->_weights(), cs);
    _hiddenToVisible = cl_float2{ fbVisibleSTDPLayer->_hiddenToVisible()->x(), fbVisibleSTDPLayer->_hiddenToVisible()->y() };
    _visibleToHidden = cl_float2{ fbVisibleSTDPLayer->_visibleToHidden()->x(), fbVisibleSTDPLayer->_visibleToHidden()->y() };
    _reverseRadii = cl_int2{ fbVisibleSTDPLayer->_reverseRadii()->x(), fbVisibleSTDPLayer->_reverseRadii()->y() };
}

flatbuffers::Offset<schemas::VisibleSTDPLayer> SparseFeaturesSTDP::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::CreateVisibleSTDPLayer(builder,
        ogmaneo::save(_derivedInput, builder, cs),
        ogmaneo::save(_weights, builder, cs),
        &hiddenToVisible, &visibleToHidden, &reverseRadii);
}

void SparseFeaturesSTDP::SparseFeaturesSTDPDesc::load(const schemas::SparseFeaturesSTDPDesc* fbSparseFeaturesSTDPDesc, ComputeSystem &cs) {
    assert(_hiddenSize.x == fbSparseFeaturesSTDPDesc->_hiddenSize()->x());
    assert(_hiddenSize.y == fbSparseFeaturesSTDPDesc->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbSparseFeaturesSTDPDesc->_visibleLayerDescs()->Length());

    _hiddenSize = cl_int2{ fbSparseFeaturesSTDPDesc->_hiddenSize()->x(), fbSparseFeaturesSTDPDesc->_hiddenSize()->y() };
    _inhibitionRadius = fbSparseFeaturesSTDPDesc->_inhibitionRadius();
    _biasAlpha = fbSparseFeaturesSTDPDesc->_biasAlpha();
    _activeRatio = fbSparseFeaturesSTDPDesc->_activeRatio();
    _gamma = fbSparseFeaturesSTDPDesc->_gamma();
    _initWeightRange = cl_float2{ fbSparseFeaturesSTDPDesc->_initWeightRange()->x(), fbSparseFeaturesSTDPDesc->_initWeightRange()->y() };

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesSTDPDesc->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSparseFeaturesSTDPDesc->_visibleLayerDescs()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::SparseFeaturesSTDPDesc> SparseFeaturesSTDP::SparseFeaturesSTDPDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);
    schemas::float2 initWeightRange(_initWeightRange.x, _initWeightRange.y);

    std::vector<schemas::VisibleSTDPLayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    return schemas::CreateSparseFeaturesSTDPDesc(builder,
        &hiddenSize, _inhibitionRadius, _biasAlpha, _activeRatio, _gamma, 
        &initWeightRange, builder.CreateVectorOfStructs(visibleLayerDescs));
}

void SparseFeaturesSTDP::load(const schemas::SparseFeatures* fbSparseFeatures, ComputeSystem &cs) {
    assert(fbSparseFeatures->_sf_type() == schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesSTDP);
    schemas::SparseFeaturesSTDP* fbSparseFeaturesSTDP =
        (schemas::SparseFeaturesSTDP*)(fbSparseFeatures->_sf());

    assert(_hiddenSize.x == fbSparseFeaturesSTDP->_hiddenSize()->x());
    assert(_hiddenSize.y == fbSparseFeaturesSTDP->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbSparseFeaturesSTDP->_visibleLayerDescs()->Length());
    assert(_visibleLayers.size() == fbSparseFeaturesSTDP->_visibleLayers()->Length());

    _hiddenSize = cl_int2{ fbSparseFeaturesSTDP->_hiddenSize()->x(), fbSparseFeaturesSTDP->_hiddenSize()->y() };

    _inhibitionRadius = fbSparseFeaturesSTDP->_inhibitionRadius();

    ogmaneo::load(_hiddenActivations, fbSparseFeaturesSTDP->_hiddenActivations(), cs);
    ogmaneo::load(_hiddenStates, fbSparseFeaturesSTDP->_hiddenStates(), cs);
    ogmaneo::load(_hiddenBiases, fbSparseFeaturesSTDP->_hiddenBiases(), cs);
    ogmaneo::load(_hiddenSummationTemp, fbSparseFeaturesSTDP->_hiddenSummationTemp(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesSTDP->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSparseFeaturesSTDP->_visibleLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesSTDP->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbSparseFeaturesSTDP->_visibleLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::SparseFeatures> SparseFeaturesSTDP::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);

    std::vector<schemas::VisibleSTDPLayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::VisibleSTDPLayer>> visibleLayers;
    for (VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    flatbuffers::Offset<schemas::SparseFeaturesSTDP> sf = schemas::CreateSparseFeaturesSTDP(builder,
        ogmaneo::save(_hiddenActivations, builder, cs),
        ogmaneo::save(_hiddenStates, builder, cs),
        ogmaneo::save(_hiddenBiases, builder, cs),
        &hiddenSize, _inhibitionRadius,
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        _biasAlpha, _activeRatio, _gamma,
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers));

    return schemas::CreateSparseFeatures(builder,
        schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesSTDP, sf.Union());
}