// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseCoder.h"

using namespace ogmaneo;

void SparseCoder::createRandom(ComputeSystem &cs, ComputeProgram &program,
    const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
    int inhibitionRadius,
    cl_float2 initWeightRange, cl_float2 initThresholdRange,
    std::mt19937 &rng)
{
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    _visibleLayerDescs = visibleLayerDescs;

    _hiddenSize = hiddenSize;

    _inhibitionRadius = inhibitionRadius;

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

        vl._derivedInput = createDoubleBuffer2D(cs, vld._size, CL_RG, CL_FLOAT);

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });

        vl._reconError = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), vld._size.x, vld._size.y);
    }

    // Hidden state data
    _hiddenStates = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenThresholds = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    _hiddenStimulusSummationTemp = createDoubleBuffer2D(cs, _hiddenSize, CL_R, CL_FLOAT);

    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    randomUniform(_hiddenThresholds[_back], cs, randomUniform2DKernel, _hiddenSize, initThresholdRange, rng);

    // Create kernels
    _stimulusKernel = cl::Kernel(program.getProgram(), "scStimulus");
    _reverseKernel = cl::Kernel(program.getProgram(), "scReverse");
    _reconstructKernel = cl::Kernel(program.getProgram(), "scReconstruct");
    _solveHiddenKernel = cl::Kernel(program.getProgram(), "scSolveHidden");
    _learnWeightsKernel = cl::Kernel(program.getProgram(), "scLearnWeights");
    _learnThresholdsKernel = cl::Kernel(program.getProgram(), "scLearnThresholds");
    _deriveInputsKernel = cl::Kernel(program.getProgram(), "scDeriveInputs");
}

void SparseCoder::activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float inputTraceDecay, float activeRatio, std::mt19937 &rng) {
    // Derive inputs
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _deriveInputsKernel.setArg(argIndex++, visibleStates[vli]);
        _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_back]);
        _deriveInputsKernel.setArg(argIndex++, vl._derivedInput[_front]);
        _deriveInputsKernel.setArg(argIndex++, inputTraceDecay);

        cs.getQueue().enqueueNDRangeKernel(_deriveInputsKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
    }

    // Start by clearing stimulus summation buffer to biases
    {
        cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
        cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

        cs.getQueue().enqueueCopyImage(_hiddenThresholds[_back], _hiddenStimulusSummationTemp[_back], zeroOrigin, zeroOrigin, hiddenRegion);
        //cs.getQueue().enqueueFillImage(_hiddenStimulusSummationTemp[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, zeroOrigin, hiddenRegion);
    }

    // Find up stimulus
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _stimulusKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _stimulusKernel.setArg(argIndex++, _hiddenStimulusSummationTemp[_back]);
            _stimulusKernel.setArg(argIndex++, _hiddenStimulusSummationTemp[_front]);
            _stimulusKernel.setArg(argIndex++, vl._weights[_back]);
            _stimulusKernel.setArg(argIndex++, vld._size);
            _stimulusKernel.setArg(argIndex++, vl._hiddenToVisible);
            _stimulusKernel.setArg(argIndex++, vld._radius);
            _stimulusKernel.setArg(argIndex++, vld._ignoreMiddle);

            cs.getQueue().enqueueNDRangeKernel(_stimulusKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
        }

        // Swap buffers
        std::swap(_hiddenStimulusSummationTemp[_front], _hiddenStimulusSummationTemp[_back]);
    }

    // Solve hidden
    {
        int argIndex = 0;

        _solveHiddenKernel.setArg(argIndex++, _hiddenStimulusSummationTemp[_back]);
        _solveHiddenKernel.setArg(argIndex++, _hiddenStates[_front]);
        _solveHiddenKernel.setArg(argIndex++, _hiddenSize);
        _solveHiddenKernel.setArg(argIndex++, _inhibitionRadius);
        _solveHiddenKernel.setArg(argIndex++, activeRatio);

        cs.getQueue().enqueueNDRangeKernel(_solveHiddenKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));
    }
}

void SparseCoder::stepEnd(ComputeSystem &cs) {
    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

    std::swap(_hiddenStates[_front], _hiddenStates[_back]);

    // Swap buffers
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        std::swap(vl._derivedInput[_front], vl._derivedInput[_back]);
    }
}

void SparseCoder::learn(ComputeSystem &cs,
    float thresholdAlpha, float activeRatio)
{
    // Reverse
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _reverseKernel.setArg(argIndex++, _hiddenStates[_front]);
            _reverseKernel.setArg(argIndex++, vl._derivedInput[_front]);
            _reverseKernel.setArg(argIndex++, vl._reconError);
            _reverseKernel.setArg(argIndex++, vl._weights[_back]);
            _reverseKernel.setArg(argIndex++, vld._size);
            _reverseKernel.setArg(argIndex++, _hiddenSize);
            _reverseKernel.setArg(argIndex++, vl._visibleToHidden);
            _reverseKernel.setArg(argIndex++, vl._hiddenToVisible);
            _reverseKernel.setArg(argIndex++, vld._radius);
            _reverseKernel.setArg(argIndex++, vl._reverseRadii);

            cs.getQueue().enqueueNDRangeKernel(_reverseKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }
    }

    // Learn weights
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        int argIndex = 0;

        _learnWeightsKernel.setArg(argIndex++, _hiddenStates[_front]);
        _learnWeightsKernel.setArg(argIndex++, _hiddenStates[_back]);
        _learnWeightsKernel.setArg(argIndex++, vl._reconError);
        _learnWeightsKernel.setArg(argIndex++, vl._weights[_back]);
        _learnWeightsKernel.setArg(argIndex++, vl._weights[_front]);
        _learnWeightsKernel.setArg(argIndex++, vld._size);
        _learnWeightsKernel.setArg(argIndex++, vl._hiddenToVisible);
        _learnWeightsKernel.setArg(argIndex++, vld._radius);
        _learnWeightsKernel.setArg(argIndex++, activeRatio);
        _learnWeightsKernel.setArg(argIndex++, vld._weightAlpha);

        cs.getQueue().enqueueNDRangeKernel(_learnWeightsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(vl._weights[_front], vl._weights[_back]);
    }

    // Bias update
    {
        int argIndex = 0;

        _learnThresholdsKernel.setArg(argIndex++, _hiddenStates[_front]);
        _learnThresholdsKernel.setArg(argIndex++, _hiddenThresholds[_back]);
        _learnThresholdsKernel.setArg(argIndex++, _hiddenThresholds[_front]);
        _learnThresholdsKernel.setArg(argIndex++, thresholdAlpha);
        _learnThresholdsKernel.setArg(argIndex++, activeRatio);

        cs.getQueue().enqueueNDRangeKernel(_learnThresholdsKernel, cl::NullRange, cl::NDRange(_hiddenSize.x, _hiddenSize.y));

        std::swap(_hiddenThresholds[_front], _hiddenThresholds[_back]);
    }
}

void SparseCoder::clearMemory(ComputeSystem &cs) {
    cl_float4 zeroColor = { 0.0f, 0.0f, 0.0f, 0.0f };

    cl::array<cl::size_type, 3> zeroOrigin = { 0, 0, 0 };
    cl::array<cl::size_type, 3> hiddenRegion = { _hiddenSize.x, _hiddenSize.y, 1 };

    // Clear buffers
    cs.getQueue().enqueueFillImage(_hiddenStates[_back], zeroColor, zeroOrigin, hiddenRegion);

    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        cs.getQueue().enqueueFillImage(vl._derivedInput[_back], zeroColor, zeroOrigin, { static_cast<cl::size_type>(vld._size.x), static_cast<cl::size_type>(vld._size.y), 1 });
    }
}

void SparseCoder::reconstruct(ComputeSystem &cs, const cl::Image2D &hiddenStates, std::vector<cl::Image2D> &reconstructions) {
    for (int vli = 0; vli < _visibleLayers.size(); vli++) {
        VisibleLayer &vl = _visibleLayers[vli];
        VisibleLayerDesc &vld = _visibleLayerDescs[vli];

        {
            int argIndex = 0;

            _reconstructKernel.setArg(argIndex++, hiddenStates);
            _reconstructKernel.setArg(argIndex++, reconstructions[vli]);
            _reconstructKernel.setArg(argIndex++, vl._weights[_back]);
            _reconstructKernel.setArg(argIndex++, vld._size);
            _reconstructKernel.setArg(argIndex++, _hiddenSize);
            _reconstructKernel.setArg(argIndex++, vl._visibleToHidden);
            _reconstructKernel.setArg(argIndex++, vl._hiddenToVisible);
            _reconstructKernel.setArg(argIndex++, vld._radius);
            _reconstructKernel.setArg(argIndex++, vl._reverseRadii);

            cs.getQueue().enqueueNDRangeKernel(_reconstructKernel, cl::NullRange, cl::NDRange(vld._size.x, vld._size.y));
        }
    }
}