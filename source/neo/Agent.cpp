// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Agent.h"

#include <assert.h>

using namespace ogmaneo;

void Agent::simStep(float reward, std::vector<ValueField2D> &inputs, bool learn) {
    // Write input
    for (int i = 0; i < _inputImages.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_inputImages[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputs[i].getSize().x), static_cast<cl::size_type>(inputs[i].getSize().y), 1 }, 0, 0, inputs[i].getData().data());

    _as.simStep(*_resources->_cs, reward, _inputImages, _inputImages, _rng, learn);

    // Get actions
    for (int i = 0; i < _actions.size(); i++)
        _resources->_cs->getQueue().enqueueReadImage(_as.getAction(i), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actions[i].getSize().x), static_cast<cl::size_type>(_actions[i].getSize().y), 1 }, 0, 0, _actions[i].getData().data());
}

void Agent::simStep(float reward, std::vector<ValueField2D> &inputs, std::vector<ValueField2D> &corruptedInputs, bool learn) {
    // Write input
    for (int i = 0; i < _inputImages.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_inputImages[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputs[i].getSize().x), static_cast<cl::size_type>(inputs[i].getSize().y), 1 }, 0, 0, inputs[i].getData().data());

    for (int i = 0; i < _inputImages.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_corruptedInputImages[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(corruptedInputs[i].getSize().x), static_cast<cl::size_type>(corruptedInputs[i].getSize().y), 1 }, 0, 0, corruptedInputs[i].getData().data());

    _as.simStep(*_resources->_cs, reward, _inputImages, _corruptedInputImages, _rng, learn);

    // Get actions
    for (int i = 0; i < _actions.size(); i++)
        _resources->_cs->getQueue().enqueueReadImage(_as.getAction(i), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_actions[i].getSize().x), static_cast<cl::size_type>(_actions[i].getSize().y), 1 }, 0, 0, _actions[i].getData().data());
}

void Agent::load(const schemas::Agent* fbAgent, ComputeSystem &cs) {
    assert(_inputImages.size() == fbAgent->_inputImages()->Length());
    assert(_corruptedInputImages.size() == fbAgent->_corruptedInputImages()->Length());
    assert(_actions.size() == fbAgent->_actions()->Length());

    _as.load(fbAgent->_as(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbAgent->_inputImages()->Length(); i++) {
        ogmaneo::load(_inputImages[i], fbAgent->_inputImages()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgent->_corruptedInputImages()->Length(); i++) {
        ogmaneo::load(_corruptedInputImages[i], fbAgent->_corruptedInputImages()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbAgent->_actions()->Length(); i++) {
        _actions[i].load(fbAgent->_actions()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::Agent> Agent::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::Image2D>> inputImages;
    for (cl::Image2D image : _inputImages)
        inputImages.push_back(ogmaneo::save(image, builder, cs));

    std::vector<flatbuffers::Offset<schemas::Image2D>> corruptedInputImages;
    for (cl::Image2D image : _corruptedInputImages)
        corruptedInputImages.push_back(ogmaneo::save(image, builder, cs));

    std::vector<flatbuffers::Offset<schemas::ValueField2D>> actions;
    for (ValueField2D values : _actions)
        actions.push_back(values.save(builder, cs));

    return schemas::CreateAgent(builder,
        _as.save(builder, cs),
        builder.CreateVector(inputImages),
        builder.CreateVector(corruptedInputImages),
        builder.CreateVector(actions));
}

void Agent::load(ComputeSystem &cs, const std::string &fileName) {
    FILE* file = fopen(fileName.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0L, SEEK_SET);
    std::vector<uint8_t> data(length);
    fread(data.data(), sizeof(uint8_t), length, file);
    fclose(file);

    flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);

    bool verified =
        schemas::VerifyAgentBuffer(verifier) |
        schemas::AgentBufferHasIdentifier(data.data());

    if (verified) {
        const schemas::Agent* agent = schemas::GetAgent(data.data());

        load(agent, cs);
    }

    return; //verified;
}

void Agent::save(ComputeSystem &cs, const std::string &fileName) {
    flatbuffers::FlatBufferBuilder builder;

    flatbuffers::Offset<schemas::Agent> agent = save(builder, cs);

    // Instruct the builder that this Agent is complete.
    schemas::FinishAgentBuffer(builder, agent);

    // Get the built buffer and size
    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier = flatbuffers::Verifier(buf, size);

    bool verified =
        schemas::VerifyAgentBuffer(verifier) |
        schemas::AgentBufferHasIdentifier(buf);

    if (verified) {
        FILE* file = fopen(fileName.c_str(), "wb");
        fwrite(buf, sizeof(uint8_t), size, file);
        fclose(file);
    }

    return; //verified;
}