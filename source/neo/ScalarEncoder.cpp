// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ScalarEncoder.h"

#include <algorithm>
#include <random>
#include <assert.h>

using namespace ogmaneo;

void ScalarEncoder::createRandom(int numInputs, int numOutputs, float initMinWeight, float initMaxWeight, int seed) {
    std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

    std::mt19937 rng(seed);

    _encoderOutputs.clear();
    _encoderOutputs.assign(numOutputs, 0.0f);

    _decoderOutputs.clear();
    _decoderOutputs.assign(numInputs, 0.0f);

    _biases.clear();
    _biases.assign(numOutputs, 0.0f);

    for (int i = 0; i < _biases.size(); i++)
        _biases[i] = weightDist(rng);

    _weightsEncode.resize(numInputs * numOutputs);
    _weightsDecode.resize(numInputs * numOutputs);

    for (int i = 0; i < _weightsEncode.size(); i++) {
        _weightsEncode[i] = weightDist(rng);
        _weightsDecode[i] = weightDist(rng);
    }
}

void ScalarEncoder::encode(const std::vector<float> &inputs, float activeRatio, float alpha, float beta) {
    assert(inputs.size() == _weightsEncode.size() / _encoderOutputs.size());

    // Compute activations
    std::vector<float> activations(_encoderOutputs.size());

    for (int i = 0; i < _encoderOutputs.size(); i++) {
        float sum = _biases[i];

        for (int j = 0; j < inputs.size(); j++) {
            float delta = inputs[j] - _weightsEncode[i * inputs.size() + j];

            sum += -delta * delta;
        }

        activations[i] = sum;
    }

    // Inhibit
    for (int i = 0; i < _encoderOutputs.size(); i++) {
        float numHigher = 0.0f;

        for (int j = 0; j < _encoderOutputs.size(); j++) {
            if (i == j)
                continue;

            if (activations[j] >= activations[i])
                numHigher++;
        }

        _encoderOutputs[i] = numHigher < activeRatio * _encoderOutputs.size() ? 1.0f : 0.0f;
    }

    std::vector<float> reconInputs(inputs.size());

    // Reconstruct
    for (int i = 0; i < inputs.size(); i++) {
        float sum = 0.0f;

        for (int j = 0; j < _encoderOutputs.size(); j++)
            sum += _weightsDecode[j * inputs.size() + i] * _encoderOutputs[j];

        reconInputs[i] = sum * activeRatio;
    }

    // Learn
    for (int i = 0; i < inputs.size(); i++) {
        for (int j = 0; j < _encoderOutputs.size(); j++)
            _weightsDecode[j * inputs.size() + i] += alpha * (inputs[i] - reconInputs[i]) * _encoderOutputs[j];
    }

    // Bias update
    for (int i = 0; i < _biases.size(); i++)
        _biases[i] += beta * (activeRatio - _encoderOutputs[i]);
}

void ScalarEncoder::decode(const std::vector<float> &outputs) {
    assert(outputs.size() == _encoderOutputs.size());

    for (int i = 0; i < _decoderOutputs.size(); i++) {
        float sum = 0.0f;
        float count = 0.0f;

        for (int j = 0; j < outputs.size(); j++) {
            sum += _weightsEncode[j * _decoderOutputs.size() + i] * outputs[j];
            count += outputs[j];
        }

        _decoderOutputs[i] = sum / fmax(1.0f, count);
    }
}