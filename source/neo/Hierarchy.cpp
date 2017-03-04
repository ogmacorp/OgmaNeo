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

void Hierarchy::activate(std::vector<ValueField2D> &inputsFeed) {
    // Write input
    for (int i = 0; i < _inputImagesFeed.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_inputImagesFeed[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputsFeed[i].getSize().x), static_cast<cl::size_type>(inputsFeed[i].getSize().y), 1 }, 0, 0, inputsFeed[i].getData().data());

    _p.activate(*_resources->_cs, _inputImagesFeed, _rng);

    // Get predictions
    for (int i = 0; i < _predictions.size(); i++) {
        _resources->_cs->getQueue().enqueueReadImage(_p.getPredictions(i)[_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(_predictions[i].getSize().x), static_cast<cl::size_type>(_predictions[i].getSize().y), 1 }, 0, 0, _predictions[i].getData().data());
    }
}

void Hierarchy::learn(std::vector<ValueField2D> &inputsPredict, float tdError) {
    // Write input
    for (int i = 0; i < _inputImagesPredict.size(); i++)
        _resources->_cs->getQueue().enqueueWriteImage(_inputImagesPredict[i], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(inputsPredict[i].getSize().x), static_cast<cl::size_type>(inputsPredict[i].getSize().y), 1 }, 0, 0, inputsPredict[i].getData().data());

    _p.learn(*_resources->_cs, _inputImagesPredict, _rng, tdError);
}

void Hierarchy::load(const schemas::Hierarchy* fbHierarchy, ComputeSystem &cs) {
    assert(_inputImagesFeed.size() == fbHierarchy->_inputImagesFeed()->Length());
    assert(_inputImagesPredict.size() == fbHierarchy->_inputImagesPredict()->Length());
    assert(_predictions.size() == fbHierarchy->_predictions()->Length());

    _p.load(fbHierarchy->_p(), cs);

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_inputImagesFeed()->Length(); i++) {
        ogmaneo::load(_inputImagesFeed[i], fbHierarchy->_inputImagesFeed()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_inputImagesPredict()->Length(); i++) {
        ogmaneo::load(_inputImagesPredict[i], fbHierarchy->_inputImagesPredict()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_predictions()->Length(); i++) {
        _predictions[i].load(fbHierarchy->_predictions()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::Hierarchy> Hierarchy::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::Image2D>> inputImagesFeed;
    for (cl::Image2D image : _inputImagesFeed)
        inputImagesFeed.push_back(ogmaneo::save(image, builder, cs));

    std::vector<flatbuffers::Offset<schemas::Image2D>> inputImagesPredict;
    for (cl::Image2D image : _inputImagesPredict)
        inputImagesPredict.push_back(ogmaneo::save(image, builder, cs));

    std::vector<flatbuffers::Offset<schemas::ValueField2D>> predictions;
    for (ValueField2D values : _predictions)
        predictions.push_back(values.save(builder, cs));

    return schemas::CreateHierarchy(builder,
        _p.save(builder, cs),
        builder.CreateVector(inputImagesFeed),
        builder.CreateVector(inputImagesPredict),
        builder.CreateVector(predictions));
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