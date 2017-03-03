// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseFeaturesDistance.h"

#include "PredictorLayer.h"

using namespace ogmaneo;

SparseFeaturesDistance::SparseFeaturesDistance(ComputeSystem &cs, ComputeProgram &sfdProgram,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
    cl_int2 chunkSize,
    float gamma,
    cl_float2 initWeightRange,
    std::mt19937 &rng)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _type = SparseFeaturesType::_distance;

    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _chunkSize = chunkSize;

    _gamma = gamma;

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    _visibleLayers.resize(_visibleLayerDescs.size());

    cl::Kernel randomUniform2DKernel = cl::Kernel(sfdProgram.getProgram(), "randomUniform2D");
    cl::Kernel randomUniform3DKernel = cl::Kernel(sfdProgram.getProgram(), "randomUniform3D");

    int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / static_cast<float>(_chunkSize.x)));
    int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / static_cast<float>(_chunkSize.y)));

    cl::array<cl::size_type, 3> chunkRegion = { static_cast<cl_uint>(chunksInX), static_cast<cl_uint>(chunksInY), 1 };

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

        vl._chunkToVisible = cl_float2{ static_cast<float>(vld._size.x) / static_cast<float>(chunksInX),
            static_cast<float>(vld._size.y) / static_cast<float>(chunksInY)
        };

        vl._reverseRadii = cl_int2{ static_cast<cl_int>(std::ceil(vl._visibleToHidden.x * vld._radius) + 1),
            static_cast<cl_int>(std::ceil(vl._visibleToHidden.y * vld._radius) + 1)
        };

        {
            int weightDiam = vld._radius * 2 + 1;

            int numWeights = weightDiam * weightDiam * vld._numSamples;

            cl_int3 weightsSize = { _hiddenSize.x, _hiddenSize.y, numWeights };

            vl._weights = createDoubleBuffer3D(cs, weightsSize, CL_R, CL_FLOAT);

            randomUniform(vl._weights[_back], cs, randomUniform3DKernel, weightsSize, { initWeightRange.x, 0.0f, 0.0f, 0.0f }, { initWeightRange.y, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, rng);
        }

        vl._derivedInputs = createDoubleBuffer2D(cs, vld._size, CL_RG, CL_FLOAT);
        cs.getQueue().enqueueFillImage(vl._derivedInputs[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });

        vl._samples = createDoubleBuffer3D(cs, { vld._size.x, vld._size.y, vld._numSamples }, CL_R, CL_FLOAT);
        cs.getQueue().enqueueFillImage(vl._samples[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(vld._numSamples) });
    
        vl._samplesAccum = createDoubleBuffer3D(cs, { vld._size.x, vld._size.y, vld._numSamples }, CL_R, CL_FLOAT);
        cs.getQueue().enqueueFillImage(vl._samplesAccum[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(vld._numSamples) });

        vl._samplesSlice = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);
    }

    // Hidden state data
    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_RG, CL_FLOAT);
    _hiddenActivations = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _chunkWinners = createDoubleBuffer2D(cs, { chunksInX, chunksInY }, CL_RG, CL_FLOAT);

    _hiddenSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenStates[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_hiddenActivations[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Create kernels
    _addSampleKernel = cl::Kernel(sfdProgram.getProgram(), "sfdAddSample");  
    _stimulusKernel = cl::Kernel(sfdProgram.getProgram(), "sfdStimulus");
    _learnWeightsKernel = cl::Kernel(sfdProgram.getProgram(), "sfdLearnWeights");
    _activateKernel = cl::Kernel(sfdProgram.getProgram(), "sfdActivate");
    _inhibitKernel = cl::Kernel(sfdProgram.getProgram(), "sfdInhibit");
    _inhibitOtherKernel = cl::Kernel(sfdProgram.getProgram(), "sfdInhibitOther");
    _deriveInputsKernel = cl::Kernel(sfdProgram.getProgram(), "sfdDeriveInputs");
    _sumKernel = cl::Kernel(sfdProgram.getProgram(), "sfdSum");
    _sliceKernel = cl::Kernel(sfdProgram.getProgram(), "sfdSlice");
}

void SparseFeaturesDistance::subSample(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, std::mt19937 &rng) {
    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Update derived inputs
        {
            int argIndex = 0;

            _deriveInputsKernel.setArg(argIndex++, visibleStates[vli]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInputs[_back]);
            _deriveInputsKernel.setArg(argIndex++, vl._derivedInputs[_front]);
            _deriveInputsKernel.setArg(argIndex++, vld._lambda);

            cs.getQueue().enqueueNDRangeKernel(_deriveInputsKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }

        // Add sample
        {
            int argIndex = 0;

            _addSampleKernel.setArg(argIndex++, vl._derivedInputs[_front]);
            _addSampleKernel.setArg(argIndex++, vl._samplesAccum[_back]);
            _addSampleKernel.setArg(argIndex++, vl._samplesAccum[_front]);
            _addSampleKernel.setArg(argIndex++, vld._numSamples);

            cs.getQueue().enqueueNDRangeKernel(_addSampleKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y)); 
        }

        std::swap(vl._derivedInputs[_front], vl._derivedInputs[_back]);
        std::swap(vl._samplesAccum[_front], vl._samplesAccum[_back]);
    }
}

cl::Image2D &SparseFeaturesDistance::getSubSample(ComputeSystem &cs, int vli, int index, std::mt19937 &rng) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    // Add sample
    {
        int argIndex = 0;

        _sliceKernel.setArg(argIndex++, vl._samples[_back]);
        _sliceKernel.setArg(argIndex++, vl._samplesSlice);
        _sliceKernel.setArg(argIndex++, index);

        cs.getQueue().enqueueNDRangeKernel(_sliceKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
    }

    return vl._samplesSlice;
}

cl::Image2D &SparseFeaturesDistance::getSubSampleAccum(ComputeSystem &cs, int vli, int index, std::mt19937 &rng) {
    VisibleLayer &vl = _visibleLayers[vli];
    VisibleLayerDesc &vld = _visibleLayerDescs[vli];

    // Add sample
    {
        int argIndex = 0;

        _sliceKernel.setArg(argIndex++, vl._samplesAccum[_back]);
        _sliceKernel.setArg(argIndex++, vl._samplesSlice);
        _sliceKernel.setArg(argIndex++, index);

        cs.getQueue().enqueueNDRangeKernel(_sliceKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
    }

    return vl._samplesSlice;
}

void SparseFeaturesDistance::activate(ComputeSystem &cs, std::mt19937 &rng) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    // Start by clearing stimulus summation buffer to 0
    cs.getQueue().enqueueFillImage(_hiddenSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);

    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Copy accumulation to samples
        cs.getQueue().enqueueCopyImage(vl._samplesAccum[_back], vl._samples[_front], zeroOrigin, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(vld._numSamples) });

        int argIndex = 0;

        _stimulusKernel.setArg(argIndex++, vl._samples[_front]);
        _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _stimulusKernel.setArg(argIndex++, _hiddenSummationTemp[_front]);
        _stimulusKernel.setArg(argIndex++, vl._weights[_back]);
        _stimulusKernel.setArg(argIndex++, vld._size);
        _stimulusKernel.setArg(argIndex++, vl._chunkToVisible);
        _stimulusKernel.setArg(argIndex++, _chunkSize);
        _stimulusKernel.setArg(argIndex++, vld._radius);
        _stimulusKernel.setArg(argIndex++, vld._numSamples);
        _stimulusKernel.setArg(argIndex++, vld._ignoreMiddle);

        cs.getQueue().enqueueNDRangeKernel(_stimulusKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        // Swap buffers
        std::swap(_hiddenSummationTemp[_front], _hiddenSummationTemp[_back]);
    }

    // Activate
    {
        int argIndex = 0;

        _activateKernel.setArg(argIndex++, _hiddenSummationTemp[_back]);
        _activateKernel.setArg(argIndex++, _hiddenStates[_back]);
        _activateKernel.setArg(argIndex++, _hiddenActivations[_front]);

        cs.getQueue().enqueueNDRangeKernel(_activateKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }

    // Inhibit
    {
        int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / static_cast<float>(_chunkSize.x)));
        int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / static_cast<float>(_chunkSize.y)));

        int argIndex = 0;

        _inhibitKernel.setArg(argIndex++, _hiddenActivations[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_back]);
        _inhibitKernel.setArg(argIndex++, _hiddenStates[_front]);
        _inhibitKernel.setArg(argIndex++, _chunkWinners[_front]);
        _inhibitKernel.setArg(argIndex++, _hiddenSize);
        _inhibitKernel.setArg(argIndex++, _chunkSize);
        _inhibitKernel.setArg(argIndex++, _gamma);

        cs.getQueue().enqueueNDRangeKernel(_inhibitKernel, cl::NullRange, cl::NDRange(chunksInX, chunksInY));
    }
}

void SparseFeaturesDistance::stepEnd(ComputeSystem &cs) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };

    std::swap(_hiddenStates[_front], _hiddenStates[_back]);
    std::swap(_hiddenActivations[_front], _hiddenActivations[_back]);
    std::swap(_chunkWinners[_front], _chunkWinners[_back]);

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        std::swap(vl._samples[_front], vl._samples[_back]);
    }
}

void SparseFeaturesDistance::learn(ComputeSystem &cs, std::mt19937 &rng) {
    // Learn weights
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        // Weight update
        {
            int argIndex = 0;

            _learnWeightsKernel.setArg(argIndex++, _chunkWinners[_back]);
            _learnWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._samples[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
            _learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
            _learnWeightsKernel.setArg(argIndex++, _hiddenSize);
            _learnWeightsKernel.setArg(argIndex++, vld._size);
            _learnWeightsKernel.setArg(argIndex++, vl._chunkToVisible);
            _learnWeightsKernel.setArg(argIndex++, _chunkSize);
            _learnWeightsKernel.setArg(argIndex++, vld._radius);
            _learnWeightsKernel.setArg(argIndex++, vld._weightAlpha);
            _learnWeightsKernel.setArg(argIndex++, vld._numSamples);

            cs.getQueue().enqueueNDRangeKernel(_learnWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        std::swap(vl._weights[_front], vl._weights[_back]);
    }
}

void SparseFeaturesDistance::inhibit(ComputeSystem &cs, const cl::Image2D &activations, cl::Image2D &states, std::mt19937 &rng) {
    // Inhibit
    int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / static_cast<float>(_chunkSize.x)));
    int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / static_cast<float>(_chunkSize.y)));

    int argIndex = 0;

    _inhibitOtherKernel.setArg(argIndex++, activations);
    _inhibitOtherKernel.setArg(argIndex++, states);
    _inhibitOtherKernel.setArg(argIndex++, _hiddenSize);
    _inhibitOtherKernel.setArg(argIndex++, _chunkSize);

    cs.getQueue().enqueueNDRangeKernel(_inhibitOtherKernel, cl::NullRange, cl::NDRange(chunksInX, chunksInY));
}

void SparseFeaturesDistance::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { static_cast<cl_uint>(_hiddenSize.x), static_cast<cl_uint>(_hiddenSize.y), 1 };
    
    int chunksInX = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.x) / static_cast<float>(_chunkSize.x)));
    int chunksInY = static_cast<int>(std::ceil(static_cast<float>(_hiddenSize.y) / static_cast<float>(_chunkSize.y)));

    cl::array<cl::size_type, 3> chunkRegion = { static_cast<cl_uint>(chunksInX), static_cast<cl_uint>(chunksInY), 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);
    cs.getQueue().enqueueFillImage(_hiddenActivations[_back], zeroColor, zeroOrigin, hiddenRegion);
 
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        cs.getQueue().enqueueFillImage(vl._samples[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(vld._numSamples) });
        cs.getQueue().enqueueFillImage(vl._samplesAccum[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), static_cast<cl::size_type>(vld._numSamples) });
    }
}

void SparseFeaturesDistance::VisibleLayerDesc::load(const schemas::VisibleDistanceLayerDesc* fbVisibleDistanceLayerDesc, ComputeSystem &cs) {
    _size = cl_int2{ fbVisibleDistanceLayerDesc->_size().x(), fbVisibleDistanceLayerDesc->_size().y() };
    _numSamples = fbVisibleDistanceLayerDesc->_numSamples();
    _radius = fbVisibleDistanceLayerDesc->_radius();
    _ignoreMiddle = fbVisibleDistanceLayerDesc->_ignoreMiddle();
    _weightAlpha = fbVisibleDistanceLayerDesc->_weightAlpha();
    _lambda = fbVisibleDistanceLayerDesc->_lambda();
}

schemas::VisibleDistanceLayerDesc SparseFeaturesDistance::VisibleLayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 size(_size.x, _size.y);
    return schemas::VisibleDistanceLayerDesc(size, _numSamples, _radius, _ignoreMiddle, _weightAlpha, _lambda);
}

void SparseFeaturesDistance::VisibleLayer::load(const schemas::VisibleDistanceLayer* fbVisibleDistanceLayer, ComputeSystem &cs) {
    ogmaneo::load(_samples, fbVisibleDistanceLayer->_samples(), cs);
    ogmaneo::load(_samplesAccum, fbVisibleDistanceLayer->_samplesAccum(), cs);
    ogmaneo::load(_weights, fbVisibleDistanceLayer->_weights(), cs);
    _hiddenToVisible = cl_float2{ fbVisibleDistanceLayer->_hiddenToVisible()->x(), fbVisibleDistanceLayer->_hiddenToVisible()->y() };
    _visibleToHidden = cl_float2{ fbVisibleDistanceLayer->_visibleToHidden()->x(), fbVisibleDistanceLayer->_visibleToHidden()->y() };
    _reverseRadii = cl_int2{ fbVisibleDistanceLayer->_reverseRadii()->x(), fbVisibleDistanceLayer->_reverseRadii()->y() };
}

flatbuffers::Offset<schemas::VisibleDistanceLayer> SparseFeaturesDistance::VisibleLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::float2 hiddenToVisible(_hiddenToVisible.x, _hiddenToVisible.y);
    schemas::float2 visibleToHidden(_visibleToHidden.x, _visibleToHidden.y);
    schemas::int2 reverseRadii(_reverseRadii.x, _reverseRadii.y);

    return schemas::CreateVisibleDistanceLayer(builder,
        ogmaneo::save(_samples, builder, cs),
        ogmaneo::save(_samplesAccum, builder, cs),
        ogmaneo::save(_weights, builder, cs),
        &hiddenToVisible, &visibleToHidden, &reverseRadii);
}

void SparseFeaturesDistance::SparseFeaturesDistanceDesc::load(const schemas::SparseFeaturesDistanceDesc* fbSparseFeaturesDistanceDesc, ComputeSystem &cs) {
    assert(_hiddenSize.x == fbSparseFeaturesDistanceDesc->_hiddenSize()->x());
    assert(_hiddenSize.y == fbSparseFeaturesDistanceDesc->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbSparseFeaturesDistanceDesc->_visibleLayerDescs()->Length());

    _hiddenSize = cl_int2{ fbSparseFeaturesDistanceDesc->_hiddenSize()->x(), fbSparseFeaturesDistanceDesc->_hiddenSize()->y() };
    _chunkSize = cl_int2{ fbSparseFeaturesDistanceDesc->_chunkSize()->x(), fbSparseFeaturesDistanceDesc->_chunkSize()->y() };
    _initWeightRange = cl_float2{ fbSparseFeaturesDistanceDesc->_initWeightRange()->x(), fbSparseFeaturesDistanceDesc->_initWeightRange()->y() };

    _gamma = fbSparseFeaturesDistanceDesc->_gamma();

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesDistanceDesc->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSparseFeaturesDistanceDesc->_visibleLayerDescs()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::SparseFeaturesDistanceDesc> SparseFeaturesDistance::SparseFeaturesDistanceDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);
    schemas::int2 chunkSize(_chunkSize.x, _chunkSize.y);
    schemas::float2 initWeightRange(_initWeightRange.x, _initWeightRange.y);

    std::vector<schemas::VisibleDistanceLayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    return schemas::CreateSparseFeaturesDistanceDesc(builder,
        &hiddenSize, &chunkSize, _gamma,
        &initWeightRange, builder.CreateVectorOfStructs(visibleLayerDescs));
}

void SparseFeaturesDistance::load(const schemas::SparseFeatures* fbSparseFeatures, ComputeSystem &cs) {
    assert(fbSparseFeatures->_sf_type() == schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesDistance);
    schemas::SparseFeaturesDistance* fbSparseFeaturesDistance =
        (schemas::SparseFeaturesDistance*)(fbSparseFeatures->_sf());

    assert(_hiddenSize.x == fbSparseFeaturesDistance->_hiddenSize()->x());
    assert(_hiddenSize.y == fbSparseFeaturesDistance->_hiddenSize()->y());
    assert(_visibleLayerDescs.size() == fbSparseFeaturesDistance->_visibleLayerDescs()->Length());
    assert(_visibleLayers.size() == fbSparseFeaturesDistance->_visibleLayers()->Length());

    _hiddenSize = cl_int2{ fbSparseFeaturesDistance->_hiddenSize()->x(), fbSparseFeaturesDistance->_hiddenSize()->y() };
    _chunkSize = cl_int2{ fbSparseFeaturesDistance->_chunkSize()->x(), fbSparseFeaturesDistance->_chunkSize()->y() };

    _gamma = fbSparseFeaturesDistance->_gamma();

    ogmaneo::load(_hiddenStates, fbSparseFeaturesDistance->_hiddenStates(), cs);
    ogmaneo::load(_hiddenActivations, fbSparseFeaturesDistance->_hiddenActivations(), cs);
    ogmaneo::load(_chunkWinners, fbSparseFeaturesDistance->_chunkWinners(), cs);
    ogmaneo::load(_hiddenSummationTemp, fbSparseFeaturesDistance->_hiddenSummationTemp(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesDistance->_visibleLayerDescs()->Length(); i++) {
        _visibleLayerDescs[i].load(fbSparseFeaturesDistance->_visibleLayerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesDistance->_visibleLayers()->Length(); i++) {
        _visibleLayers[i].load(fbSparseFeaturesDistance->_visibleLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::SparseFeatures> SparseFeaturesDistance::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::int2 hiddenSize(_hiddenSize.x, _hiddenSize.y);
    schemas::int2 chunkSize(_chunkSize.x, _chunkSize.y);

    std::vector<schemas::VisibleDistanceLayerDesc> visibleLayerDescs;
    for (VisibleLayerDesc layerDesc : _visibleLayerDescs)
        visibleLayerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::VisibleDistanceLayer>> visibleLayers;
    for (VisibleLayer layer : _visibleLayers)
        visibleLayers.push_back(layer.save(builder, cs));

    flatbuffers::Offset<schemas::SparseFeaturesDistance> sf = schemas::CreateSparseFeaturesDistance(builder,
        ogmaneo::save(_hiddenStates, builder, cs),
        ogmaneo::save(_hiddenActivations, builder, cs),
        ogmaneo::save(_chunkWinners, builder, cs),
        &hiddenSize, &chunkSize, _gamma,
        ogmaneo::save(_hiddenSummationTemp, builder, cs),
        builder.CreateVectorOfStructs(visibleLayerDescs),
        builder.CreateVector(visibleLayers));

    return schemas::CreateSparseFeatures(builder,
        schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesDistance, sf.Union());
}
