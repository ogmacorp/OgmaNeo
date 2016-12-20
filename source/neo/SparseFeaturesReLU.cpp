// --------------------------------------------------------------------------
//	Ogma Toolkit(OTK)
//	Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
// --------------------------------------------------------------------------

#include "SparseFeaturesReLU.h"

using namespace ogmaneo;

SparseFeaturesReLU::SparseFeaturesReLU(ComputeSystem &cs, ComputeProgram &sfrProgram,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
    int numSamples, int lateralRadius,
    cl_float gamma, cl_float activeRatio, cl_float biasAlpha,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
    : _lateralRadius(lateralRadius), _gamma(gamma), _activeRatio(activeRatio), _biasAlpha(biasAlpha)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _type = SparseFeaturesType::_ReLU;

    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _numSamples = numSamples;

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    _visibleLayers.resize(_visibleLayerDescs.size());

    cl::Kernel randomUniform2DKernel = cl::Kernel(sfrProgram.getProgram(), "randomUniform2D");
    cl::Kernel randomUniform3DKernel = cl::Kernel(sfrProgram.getProgram(), "randomUniform3D");

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

        vl._reverseRadiiHidden = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radiusHidden) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radiusHidden) + 1)
        };

        vl._reverseRadiiVisible = cl_int2{ static_cast<cl_int>(std::ceil(vl._hiddenToVisible.x * vld._radiusVisible) + 1),
            static_cast<cl_int>(std::ceil(vl._hiddenToVisible.y * vld._radiusVisible) + 1)
        };

        {
            int weightDiam = vld._radiusHidden * 2 + 1;

            int numWeights = weightDiam * weightDiam * _numSamples;

            cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

            vl._weightsHidden = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

            randomUniform(vl._weightsHidden[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }

        if (vld._predict) {
            int weightDiam = vld._radiusVisible * 2 + 1;

            int numWeights = weightDiam * weightDiam;

            cl_int3 weightsSize = { vld._size.x, vld._size.y, numWeights };

            vl._weightsVisible = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

            randomUniform(vl._weightsVisible[_back], cs, randomUniform3DKernel, weightsSize, initWeightRange, rng);
        }

        vl._derivedInput = createDoubleBuffer2D(cs, vld._size, CL_RG, CL_FLOAT);
        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });

        if (vld._predict) {
            vl._predictions = createDoubleBuffer2D(cs, vld._size, CL_R, CL_FLOAT);
            cs.getQueue().enqueueFillImage(vl._predictions[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
        }

        vl._samples = createDoubleBuffer3D(cs, { vld._size.x, vld._size.y, numSamples }, CL_R, CL_FLOAT);
        cs.getQueue().enqueueFillImage(vl._samples[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(numSamples) });
    }

    // Hidden state data
    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);
    _hiddenBiases = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenStates[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    randomUniform(_hiddenBiases[_back], cs, randomUniform2DKernel, _hiddenSize, initWeightRange, rng);

    // Create kernels
    _addSampleKernel = cl::Kernel(sfrProgram.getProgram(), "sfrAddSample");
    _stimulusKernel = cl::Kernel(sfrProgram.getProgram(), "sfrStimulus");
    _inhibitKernel = cl::Kernel(sfrProgram.getProgram(), "sfrInhibit");
    _predictKernel = cl::Kernel(sfrProgram.getProgram(), "sfrPredict");
    _inhibitOtherKernel = cl::Kernel(sfrProgram.getProgram(), "sfrInhibitOther");
    _errorPropKernel = cl::Kernel(sfrProgram.getProgram(), "sfrErrorProp");
    _learnWeightsHiddenKernel = cl::Kernel(sfrProgram.getProgram(), "sfrLearnWeightsHidden");
    _learnWeightsVisibleKernel = cl::Kernel(sfrProgram.getProgram(), "sfrLearnWeightsVisible");
    _learnBiasesKernel = cl::Kernel(sfrProgram.getProgram(), "sfrLearnBiases");
    _deriveInputsKernel = cl::Kernel(sfrProgram.getProgram(), "sfrDeriveInputs");
}

void SparseFeaturesReLU::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, const cl::Image2D &predictionsPrev, std::mt19937 &rng) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Start by clearing stimulus summation buffer to biases
    cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);
    //cs.getQueue().enqueueCopyImage(_hiddenBiases[_back], _hiddenSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);

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

        // Add sample
        {
            int argIndex = 0;

            _addSampleKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _addSampleKernel.setArg(argIndex++, vl._samples[_back]);
            _addSampleKernel.setArg(argIndex++, vl._samples[_front]);
            _addSampleKernel.setArg(argIndex++, _numSamples);

            cs.getQueue().enqueueNDRangeKernel(_addSampleKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }

        {
            int argIndex = 0;

            _stimulusKernel.setArg(argIndex++, vl._samples[_front]);
            _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
            _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
            _stimulusKernel.setArg(argIndex++, vl._weightsHidden[_back]);
            _stimulusKernel.setArg(argIndex++, vld._size);
            _stimulusKernel.setArg(argIndex++, vl._hiddenToVisible);
            _stimulusKernel.setArg(argIndex++, vld._radiusHidden);
            _stimulusKernel.setArg(argIndex++, _numSamples);
            _stimulusKernel.setArg(argIndex++, vld._ignoreMiddle);

            cs.getQueue().enqueueNDRangeKernel(_stimulusKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
    }

    // Inhibit
    {
        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_back]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);
        _inhibitKernel.setArg(argIndex++, _lateralRadius);
        _inhibitKernel.setArg(argIndex++, _activeRatio);
        _inhibitKernel.setArg(argIndex++, _gamma);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Predict
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        if (vld._predict) {
            int argIndex = 0;

            _predictKernel.setArg(argIndex++, _hiddenStates[_front]);
            _predictKernel.setArg(argIndex++, vl._weightsVisible[_back]);
            _predictKernel.setArg(argIndex++, vl._predictions[_front]);
            _predictKernel.setArg(argIndex++, _hiddenSize);
            _predictKernel.setArg(argIndex++, vl._visibleToHidden);
            _predictKernel.setArg(argIndex++, vld._radiusVisible);

            cs.getQueue().enqueueNDRangeKernel(_predictKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }
    }
}

void SparseFeaturesReLU::stepEnd(ComputeSystem &cs) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    std::swap(_hiddenStates[_front], _hiddenStates[_back]);

    // Swap buffers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
        std::swap(vl._samples[_front], vl._samples[_back]);
        std::swap(vl._predictions[_front], vl._predictions[_back]);
    }
}

void SparseFeaturesReLU::learn(ComputeSystem &cs, const cl::Image2D &predictionsPrev, std::mt19937 &rng) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Propagate errors
    cs.getQueue().enqueueCopyImage(_hiddenBiases[_back], _hiddenSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);

    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        if (vld._predict) {
            {
                int argIndex = 0;

                _errorPropKernel.setArg(argIndex++, vl._derivedInput[_front]);
                _errorPropKernel.setArg(argIndex++, vl._predictions[_back]);
                _errorPropKernel.setArg(argIndex++, _hiddenStates[_front]);
                _errorPropKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
                _errorPropKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
                _errorPropKernel.setArg(argIndex++, vl._weightsVisible[_back]);
                _errorPropKernel.setArg(argIndex++, vld._size);
                _errorPropKernel.setArg(argIndex++, _hiddenSize);
                _errorPropKernel.setArg(argIndex++, vl._visibleToHidden);
                _errorPropKernel.setArg(argIndex++, vl._hiddenToVisible);
                _errorPropKernel.setArg(argIndex++, vld._radiusVisible);
                _errorPropKernel.setArg(argIndex++, vl._reverseRadiiVisible);

                cs.getQueue().enqueueNDRangeKernel(_errorPropKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

                std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
            }

            // While here, learn visible weights
            {
                int argIndex = 0;

                _learnWeightsVisibleKernel.setArg(argIndex++, _hiddenStates[_front]);
                _learnWeightsVisibleKernel.setArg(argIndex++, vl._weightsVisible[_back]);
                _learnWeightsVisibleKernel.setArg(argIndex++, vl._weightsVisible[_front]);
                _learnWeightsVisibleKernel.setArg(argIndex++, vl._derivedInput[_front]);
                _learnWeightsVisibleKernel.setArg(argIndex++, vl._predictions[_back]);
                _learnWeightsVisibleKernel.setArg(argIndex++, _hiddenSize);
                _learnWeightsVisibleKernel.setArg(argIndex++, vl._visibleToHidden);
                _learnWeightsVisibleKernel.setArg(argIndex++, vld._radiusVisible);
                _learnWeightsVisibleKernel.setArg(argIndex++, vld._weightAlphaVisible);

                cs.getQueue().enqueueNDRangeKernel(_learnWeightsVisibleKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));

                std::swap(vl._weightsVisible[_front], vl._weightsVisible[_back]);
            }
        }
    }

    // Learn weights
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Weight update
        {
            int argIndex = 0;

            _learnWeightsHiddenKernel.setArg(argIndex++, _hiddenStates[_back]);
            _learnWeightsHiddenKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
            _learnWeightsHiddenKernel.setArg(argIndex++, vl._samples[_back]);
            _learnWeightsHiddenKernel.setArg(argIndex++, vl._weightsHidden[_back]);
            _learnWeightsHiddenKernel.setArg(argIndex++, vl._weightsHidden[_front]);
            _learnWeightsHiddenKernel.setArg(argIndex++, vld._size);
            _learnWeightsHiddenKernel.setArg(argIndex++, vl._hiddenToVisible);
            _learnWeightsHiddenKernel.setArg(argIndex++, vld._radiusHidden);
            _learnWeightsHiddenKernel.setArg(argIndex++, vld._weightAlphaHidden);
            _learnWeightsHiddenKernel.setArg(argIndex++, _numSamples);
            _learnWeightsHiddenKernel.setArg(argIndex++, _activeRatio);

            cs.getQueue().enqueueNDRangeKernel(_learnWeightsHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        std::swap(vl._weightsHidden[_front], vl._weightsHidden[_back]);
    }

    {
        int argIndex = 0;

        _learnBiasesKernel.setArg(argIndex++, _hiddenStates[_back]);
        _learnBiasesKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_back]);
        _learnBiasesKernel.setArg(argIndex++, _hiddenBiases[_front]);
        _learnBiasesKernel.setArg(argIndex++, _activeRatio);
        _learnBiasesKernel.setArg(argIndex++, _biasAlpha);

        cs.getQueue().enqueueNDRangeKernel(_learnBiasesKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(_hiddenBiases[_front], _hiddenBiases[_back]);
    }
}

void SparseFeaturesReLU::inhibit(ComputeSystem &cs, const cl::Image2D &activations, cl::Image2D &states, std::mt19937 &rng) {
    // Inhibit
    {
        int argIndex = 0;

        _inhibitOtherKernel.setArg(argIndex++, activations);
        _inhibitOtherKernel.setArg(argIndex++, states);
        _inhibitOtherKernel.setArg(argIndex++, _hiddenSize);
        _inhibitOtherKernel.setArg(argIndex++, _lateralRadius);
        _inhibitOtherKernel.setArg(argIndex++, _activeRatio);

        cs.getQueue().enqueueNDRangeKernel(_inhibitOtherKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void SparseFeaturesReLU::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
        cs.getQueue().enqueueFillImage(vl._samples[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(_numSamples) });
        cs.getQueue().enqueueFillImage(vl._predictions[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(_numSamples) });
    }
}

void SparseFeaturesReLU::VisibleLayerDesc::load(const schemas::VisibleReLULayerDesc* fbVisibleReLULayerDesc, ComputeSystem &cs) {
    _size = cl_int2{ fbVisibleReLULayerDesc->_size().x(), fbVisibleReLULayerDesc->_size().y() };
    _radiusHidden = fbVisibleReLULayerDesc->_radiusHidden();
    _radiusVisible = fbVisibleReLULayerDesc->_radiusVisible();
    _ignoreMiddle = fbVisibleReLULayerDesc->_ignoreMiddle();
    _weightAlphaHidden = fbVisibleReLULayerDesc->_weightAlphaHidden();
    _weightAlphaVisible = fbVisibleReLULayerDesc->_weightAlphaVisible();
    _lambda = fbVisibleReLULayerDesc->_lambda();
    _predict = fbVisibleReLULayerDesc->_predict();
}

schemas::VisibleReLULayerDesc SparseFeaturesReLU::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 size(_size.x, _size.y);
    return schemas::VisibleReLULayerDesc(size, _radiusHidden, _radiusVisible, _ignoreMiddle, _weightAlphaHidden, _weightAlphaVisible, _lambda, _predict);
}

void SparseFeaturesReLU::VisibleLayer::load(const schemas::VisibleReLULayer* fbVisibleReLULayer, ComputeSystem &cs) {
    ogmaneo::load(_derivedInput, fbVisibleReLULayer->_derivedInput(), cs);
    ogmaneo::load(_predictions, fbVisibleReLULayer->_predictions(), cs);
    ogmaneo::load(_samples, fbVisibleReLULayer->_samples(), cs);
    ogmaneo::load(_weightsHidden, fbVisibleReLULayer->_weightsHidden(), cs);
    ogmaneo::load(_weightsVisible, fbVisibleReLULayer->_weightsVisible(), cs);
    _hiddenToVisible = cl_float2{ fbVisibleReLULayer->_hiddenToVisible()->x(), fbVisibleReLULayer->_hiddenToVisible()->y() };
    _visibleToHidden = cl_float2{ fbVisibleReLULayer->_visibleToHidden()->x(), fbVisibleReLULayer->_visibleToHidden()->y() };
    _reverseRadiiHidden = cl_int2{ fbVisibleReLULayer->_reverseRadiiHidden()->x(), fbVisibleReLULayer->_reverseRadiiHidden()->y() };
    _reverseRadiiVisible = cl_int2{ fbVisibleReLULayer->_reverseRadiiVisible()->x(), fbVisibleReLULayer->_reverseRadiiVisible()->y() };
}

flatbuffers::Offset<schemas::VisibleReLULayer> SparseFeaturesReLU::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadiiHidden(_reverseRadiiHidden.x, _reverseRadiiHidden.y);
    schemas::int2 reverseRadiiVisible(_reverseRadiiVisible.x, _reverseRadiiVisible.y);

    return schemas::CreateVisibleReLULayer(builder,
        ogmaneo::save(_derivedInput, builder, cs),
        ogmaneo::save(_predictions, builder, cs),
        ogmaneo::save(_samples, builder, cs),
        ogmaneo::save(_weightsHidden, builder, cs),
        ogmaneo::save(_weightsVisible, builder, cs),
        &hiddenToVisible, &visibleToHidden, &reverseRadiiHidden, &reverseRadiiVisible);
}

void SparseFeaturesReLU::SparseFeaturesReLUDesc::load(const schemas::SparseFeaturesReLUDesc* fbSparseFeaturesReLUDesc, ComputeSystem &cs) {
    assert(_hiddenSize.x == fbSparseFeaturesReLUDesc->_hiddenSize()->x());
    assert(_hiddenSize.y == fbSparseFeaturesReLUDesc->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbSparseFeaturesReLUDesc->_visibleLayerDescs()->Length());

    _hiddenSize = cl_int2{ fbSparseFeaturesReLUDesc->_hiddenSize()->x(), fbSparseFeaturesReLUDesc->_hiddenSize()->y() };
    _numSamples = fbSparseFeaturesReLUDesc->_numSamples();
    _lateralRadius = fbSparseFeaturesReLUDesc->_lateralRadius();
    _gamma = fbSparseFeaturesReLUDesc->_gamma();
    _activeRatio = fbSparseFeaturesReLUDesc->_activeRatio();
    _biasAlpha = fbSparseFeaturesReLUDesc->_biasAlpha();
    _initWeightRange = cl_float2{ fbSparseFeaturesReLUDesc->_initWeightRange()->x(), fbSparseFeaturesReLUDesc->_initWeightRange()->y() };

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesReLUDesc->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSparseFeaturesReLUDesc->_visibleLayerDescs()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::SparseFeaturesReLUDesc> SparseFeaturesReLU::SparseFeaturesReLUDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);
    schemas::float2 initWeightRange(_initWeightRange.x, _initWeightRange.y);

    std::vector<schemas::VisibleReLULayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    return schemas::CreateSparseFeaturesReLUDesc(builder,
        &hiddenSize, _numSamples, _lateralRadius, _gamma, _activeRatio, _biasAlpha,
        &initWeightRange, builder.CreateVectorOfStructs(visibleLayerDescs));
}

void SparseFeaturesReLU::load(const schemas::SparseFeatures* fbSparseFeatures, ComputeSystem &cs) {
    assert(fbSparseFeatures->_sf_type() == schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesReLU);
    schemas::SparseFeaturesReLU* fbSparseFeaturesReLU =
        (schemas::SparseFeaturesReLU*)(fbSparseFeatures->_sf());

    assert(_hiddenSize.x == fbSparseFeaturesReLU->_hiddenSize()->x());
    assert(_hiddenSize.y == fbSparseFeaturesReLU->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbSparseFeaturesReLU->_visibleLayerDescs()->Length());
    assert(_visibleLayers.size() == fbSparseFeaturesReLU->_visibleLayers()->Length());

    _hiddenSize = cl_int2{ fbSparseFeaturesReLU->_hiddenSize()->x(), fbSparseFeaturesReLU->_hiddenSize()->y() };
    _numSamples = fbSparseFeaturesReLU->_numSamples();
    _lateralRadius = fbSparseFeaturesReLU->_lateralRadius();
    _gamma = fbSparseFeaturesReLU->_gamma();
    _activeRatio = fbSparseFeaturesReLU->_activeRatio();
    _biasAlpha = fbSparseFeaturesReLU->_biasAlpha();

    ogmaneo::load(_hiddenStates, fbSparseFeaturesReLU->_hiddenStates(), cs);
    ogmaneo::load(_hiddenBiases, fbSparseFeaturesReLU->_hiddenBiases(), cs);
    ogmaneo::load(_hiddenSummationTemp, fbSparseFeaturesReLU->_hiddenSummationTemp(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesReLU->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSparseFeaturesReLU->_visibleLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesReLU->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbSparseFeaturesReLU->_visibleLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::SparseFeatures> SparseFeaturesReLU::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);

    std::vector<schemas::VisibleReLULayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::VisibleReLULayer>> visibleLayers;
    for (VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    flatbuffers::Offset<schemas::SparseFeaturesReLU> sf = schemas::CreateSparseFeaturesReLU(builder,
        ogmaneo::save(_hiddenStates, builder, cs),
        ogmaneo::save(_hiddenBiases, builder, cs),
        &hiddenSize,
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        _numSamples, _lateralRadius, _gamma, _activeRatio, _biasAlpha,
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers));

    return schemas::CreateSparseFeatures(builder,
        schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesReLU, sf.Union());
}