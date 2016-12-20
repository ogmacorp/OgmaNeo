// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <assert.h>

using namespace ogmaneo;

void Hierarchy::simStep(std::vector<ValueField2D> &inputs, bool learn) {
    // Write input
    for (int i = 0; i < _inputImages.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_inputImages[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputs[i].getSize().x), static_cast<cl::size_type>(inputs[i].getSize().y), 1 }, 0, 0, inputs[i].getData().data());

    _p.simStep(*_resources->_cs, _inputImages, _inputImages, _rng, learn);

    // Get predictions
    for (int i = 0; i < _predictions.size(); i++) {
        _readoutLayers[i].activate(*_resources->_cs, { _p.getHiddenPrediction()[_back] }, _rng);

        if (learn)
            _readoutLayers[i].learn(*_resources->_cs, _inputImages[i]);

        _readoutLayers[i].stepEnd(*_resources->_cs);

        _resources->_cs->getQueue().enqueueReadImage(_readoutLayers[i].getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_predictions[i].getSize().x), static_cast<cl::size_type>(_predictions[i].getSize().y), 1 }, 0, 0, _predictions[i].getData().data());
    }
}

void Hierarchy::simStep(std::vector<ValueField2D> &inputs, std::vector<ValueField2D> &corruptedInputs, bool learn) {
    // Write input
    for (int i = 0; i < _inputImages.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_inputImages[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputs[i].getSize().x), static_cast<cl::size_type>(inputs[i].getSize().y), 1 }, 0, 0, inputs[i].getData().data());

    for (int i = 0; i < _corruptedInputImages.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_corruptedInputImages[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(corruptedInputs[i].getSize().x), static_cast<cl::size_type>(corruptedInputs[i].getSize().y), 1 }, 0, 0, corruptedInputs[i].getData().data());

    _p.simStep(*_resources->_cs, _inputImages, _corruptedInputImages, _rng, learn);

    // Get predictions
    for (int i = 0; i < _predictions.size(); i++) {
        _readoutLayers[i].activate(*_resources->_cs, { _p.getHiddenPrediction()[_back] }, _rng);

        if (learn)
            _readoutLayers[i].learn(*_resources->_cs, _inputImages[i]);

        _readoutLayers[i].stepEnd(*_resources->_cs);

        _resources->_cs->getQueue().enqueueReadImage(_readoutLayers[i].getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_predictions[i].getSize().x), static_cast<cl::size_type>(_predictions[i].getSize().y), 1 }, 0, 0, _predictions[i].getData().data());
    }
}

void Hierarchy::load(const schemas::Hierarchy* fbHierarchy, ComputeSystem &cs) {
    assert(_inputImages.size() == fbHierarchy->_inputImages()->Length());
    assert(_corruptedInputImages.size() == fbHierarchy->_corruptedInputImages()->Length());
    assert(_predictions.size() == fbHierarchy->_predictions()->Length());
    assert(_readoutLayers.size() == fbHierarchy->_readoutLayers()->Length());

    _p.load(fbHierarchy->_p(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_inputImages()->Length(); i++) {
        ogmaneo::load(_inputImages[i], fbHierarchy->_inputImages()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_corruptedInputImages()->Length(); i++) {
        ogmaneo::load(_corruptedInputImages[i], fbHierarchy->_corruptedInputImages()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_predictions()->Length(); i++) {
        _predictions[i].load(fbHierarchy->_predictions()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_readoutLayers()->Length(); i++) {
        _readoutLayers[i].load(fbHierarchy->_readoutLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::Hierarchy> Hierarchy::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::Image2D>> inputImages;
    for (cl::Image2D image : _inputImages)
        inputImages.push_back(ogmaneo::save(image, builder, cs));

    std::vector<flatbuffers::Offset<schemas::Image2D>> corruptedInputImages;
    for (cl::Image2D image : _corruptedInputImages)
        corruptedInputImages.push_back(ogmaneo::save(image, builder, cs));

    std::vector<flatbuffers::Offset<schemas::ValueField2D>> predictions;
    for (ValueField2D values : _predictions)
        predictions.push_back(values.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::PredictorLayer>> readoutLayers;
    for (PredictorLayer layer : _readoutLayers)
        readoutLayers.push_back(layer.save(builder, cs));

    return schemas::CreateHierarchy(builder,
        _p.save(builder, cs),
        builder.CreateVector(inputImages),
        builder.CreateVector(corruptedInputImages),
        builder.CreateVector(predictions),
        builder.CreateVector(readoutLayers));
}

void Hierarchy::load(ComputeSystem &cs, const std::string &fileName) {
    FILE* file = fopen(fileName.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0L, SEEK_SET);
    std::vector<uint8_t> data(length);
    fread(data.data(), sizeof(uint8_t), length, file);
    fclose(file);

    flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);

    bool verified =
        schemas::VerifyHierarchyBuffer(verifier) |
        schemas::HierarchyBufferHasIdentifier(data.data());

    if (verified) {
        const schemas::Hierarchy* h = schemas::GetHierarchy(data.data());

        load(h, cs);
    }

    return; //verified;
}

void Hierarchy::save(ComputeSystem &cs, const std::string &fileName) {
    flatbuffers::FlatBufferBuilder builder;

    flatbuffers::Offset<schemas::Hierarchy> h = save(builder, cs);

    // Instruct the builder that this Hierarchy is complete.
    schemas::FinishHierarchyBuffer(builder, h);

    // Get the built buffer and size
    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier = flatbuffers::Verifier(buf, size);

    bool verified =
        schemas::VerifyHierarchyBuffer(verifier) |
        schemas::HierarchyBufferHasIdentifier(buf);

    if (verified) {
        FILE* file = fopen(fileName.c_str(), "wb");
        fwrite(buf, sizeof(uint8_t), size, file);
        fclose(file);
    }

    return; //verified;
}

void Hierarchy::readChunkStates(int li, ValueField2D &valueField) {
    assert(getPredictor().getHierarchy().getLayer(li)._sf->_type == _chunk);

    valueField = ValueField2D(ogmaneo::Vec2i(getPredictor().getHierarchy().getLayer(li)._sf->getHiddenSize().x, getPredictor().getHierarchy().getLayer(li)._sf->getHiddenSize().y));

    _resources->getComputeSystem()->getQueue().enqueueReadImage(getPredictor().getHierarchy().getLayer(li)._sf->getHiddenStates()[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(getPredictor().getHierarchy().getLayer(li)._sf->getHiddenSize().x), static_cast<cl::size_type>(getPredictor().getHierarchy().getLayer(li)._sf->getHiddenSize().y), 1 }, 0, 0, valueField.getData().data());
}