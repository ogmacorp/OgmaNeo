// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Architect.h"

#include "Hierarchy.h"

// Encoders
#include "SparseFeaturesChunk.h"
#include "SparseFeaturesDistance.h"

#include <iostream>

using namespace ogmaneo;

const std::string ogmaneo::ParameterModifier::_boolTrue = "true";
const std::string ogmaneo::ParameterModifier::_boolFalse = "false";

void Architect::initialize(unsigned int seed, const std::shared_ptr<Resources> &resources) {
    _rng.seed(seed);

    _resources = resources;
}

ParameterModifier Architect::addInputLayer(const Vec2i &size, bool isQ, Vec2i chunkSize) {
    InputLayer inputLayer;
    inputLayer._size = size;
    inputLayer._isQ = isQ;
    inputLayer._chunkSize = chunkSize;

    _inputLayers.push_back(inputLayer);

    ParameterModifier pm;

    pm._target = &_inputLayers.back()._params;

    return pm;
}

ParameterModifier Architect::addHigherLayer(const Vec2i &size, SparseFeaturesType type) {
    HigherLayer higherLayer;
    higherLayer._size = size;
    higherLayer._type = type;

    _higherLayers.push_back(higherLayer);

    ParameterModifier pm;

    pm._target = &_higherLayers.back()._params;

    return pm;
}

std::shared_ptr<class Hierarchy> Architect::generateHierarchy(std::unordered_map<std::string, std::string> &additionalParams) {
    std::shared_ptr<Hierarchy> h = std::make_shared<Hierarchy>();

    h->_rng = _rng;
    h->_resources = _resources;

    h->_inputImagesFeed.resize(_inputLayers.size());
    h->_inputImagesPredict.resize(_inputLayers.size());

    std::vector<bool> shouldPredict(_inputLayers.size());

    for (int i = 0; i < _inputLayers.size(); i++) {
        h->_inputImagesFeed[i] = cl::Image2D(_resources->_cs->getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputLayers[i]._size.x, _inputLayers[i]._size.y);
        h->_inputImagesPredict[i] = cl::Image2D(_resources->_cs->getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _inputLayers[i]._size.x, _inputLayers[i]._size.y);

        /*if (_inputLayers[i]._params.find("in_predict") != _inputLayers[i]._params.end()) {
        if (_inputLayers[i]._params["in_predict"] == ParameterModifier::_boolTrue) {
        h->_predictions.push_back(ValueField2D(_inputLayers[i]._size));

        shouldPredict[i] = true;
        }
        else
        shouldPredict[i] = false;
        }
        else {*/
        h->_predictions.push_back(ValueField2D(_inputLayers[i]._size));

        shouldPredict[i] = true;
        //}
    }

    std::shared_ptr<ComputeProgram> hProg;

    if (_resources->_programs.find("hierarchy") == _resources->_programs.end()) {
        hProg = std::make_shared<ComputeProgram>();
        hProg->loadHierarchyKernel(*_resources->_cs);
    }
    else
        hProg = _resources->_programs["hierarchy"];

    std::shared_ptr<ComputeProgram> pProg;

    if (_resources->_programs.find("predictor") == _resources->_programs.end()) {
        pProg = std::make_shared<ComputeProgram>();
        pProg->loadPredictorKernel(*_resources->_cs);
    }
    else
        pProg = _resources->_programs["predictor"];

    std::vector<std::vector<Predictor::PredLayerDesc>> pLayerDescs(_higherLayers.size());
    std::vector<FeatureHierarchy::LayerDesc> hLayerDescs(_higherLayers.size());

    cl_float2 initWeightRange = { -0.01f, 0.01f };

    if (additionalParams.find("ad_initWeightRange") != additionalParams.end()) {
        Vec2f range = ParameterModifier::parseVec2f(additionalParams["ad_initWeightRange"]);

        initWeightRange = { range.x, range.y };
    }

    // Fill out layer descs
    for (int l = 0; l < _higherLayers.size(); l++) {
        if (_higherLayers[l]._params.find("hl_poolSteps") != _higherLayers[l]._params.end())
            hLayerDescs[l]._poolSteps = std::stoi(_higherLayers[l]._params["hl_poolSteps"]);

        hLayerDescs[l]._sfDesc = sfDescFromName(l, _higherLayers[l]._type, _higherLayers[l]._size, SparseFeatures::_feedForwardRecurrent, _higherLayers[l]._params);

        // P layer desc
        pLayerDescs[l].resize(l == 0 ? _inputLayers.size() : hLayerDescs[l - 1]._poolSteps);

        for (int k = 0; k < pLayerDescs[l].size(); k++) {
            if (l == 0)
                pLayerDescs[l][k]._isQ = _inputLayers[k]._isQ;

            if (_higherLayers[l]._params.find("p_alpha") != _higherLayers[l]._params.end())
                pLayerDescs[l][k]._alpha = std::stof(_higherLayers[l]._params["p_alpha"]);

            if (_higherLayers[l]._params.find("p_beta") != _higherLayers[l]._params.end())
                pLayerDescs[l][k]._beta = std::stof(_higherLayers[l]._params["p_beta"]);

            if (_higherLayers[l]._params.find("p_lambda") != _higherLayers[l]._params.end())
                pLayerDescs[l][k]._lambda = std::stof(_higherLayers[l]._params["p_lambda"]);

            if (_higherLayers[l]._params.find("p_radius") != _higherLayers[l]._params.end())
                pLayerDescs[l][k]._radius = std::stoi(_higherLayers[l]._params["p_radius"]);
        }
    }

    std::vector<cl_int2> inputSizes(_inputLayers.size());
    std::vector<cl_int2> inputChunkSizes(_inputLayers.size());

    for (int i = 0; i < _inputLayers.size(); i++) {
        inputSizes[i] = cl_int2{ _inputLayers[i]._size.x, _inputLayers[i]._size.y };
        inputChunkSizes[i] = cl_int2{ _inputLayers[i]._chunkSize.x, _inputLayers[i]._chunkSize.y };
    }

    h->_p.createRandom(*_resources->_cs, *hProg, *pProg, inputSizes, inputChunkSizes, pLayerDescs, hLayerDescs, initWeightRange, _rng);

    return h;
}

std::shared_ptr<SparseFeatures::SparseFeaturesDesc> Architect::sfDescFromName(int layerIndex, SparseFeaturesType type, const Vec2i &size,
    SparseFeatures::InputType inputType, std::unordered_map<std::string, std::string> &params)
{
    std::shared_ptr<SparseFeatures::SparseFeaturesDesc> sfDesc;

    switch (type) {
    case _chunk:
    {
        std::shared_ptr<SparseFeaturesChunk::SparseFeaturesChunkDesc> sfDescChunk = std::make_shared<SparseFeaturesChunk::SparseFeaturesChunkDesc>();

        sfDescChunk->_cs = _resources->_cs;
        sfDescChunk->_inputType = SparseFeatures::_feedForward;
        sfDescChunk->_hiddenSize = { size.x, size.y };
        sfDescChunk->_rng = _rng;

        if (params.find("sfc_chunkSize") != params.end()) {
            Vec2i chunkSize = ParameterModifier::parseVec2i(params["sfc_chunkSize"]);
            sfDescChunk->_chunkSize = { chunkSize.x, chunkSize.y };
        }

        if (params.find("sfc_gamma") != params.end())
            sfDescChunk->_gamma = std::stof(params["sfc_gamma"]);

        if (params.find("sfc_initWeightRange") != params.end()) {
            Vec2f initWeightRange = ParameterModifier::parseVec2f(params["sfc_initWeightRange"]);
            sfDescChunk->_initWeightRange = { initWeightRange.x, initWeightRange.y };
        }

        if (_resources->_programs.find("chunk") == _resources->_programs.end()) {
            _resources->_programs["chunk"] = sfDescChunk->_sfcProgram = std::make_shared<ComputeProgram>();

            sfDescChunk->_sfcProgram->loadSparseFeaturesKernel(*_resources->_cs, _chunk);
        }
        else
            sfDescChunk->_sfcProgram = _resources->_programs["chunk"];

        if (layerIndex == 0) {
            sfDescChunk->_visibleLayerDescs.resize(_inputLayers.size());

            for (int i = 0; i < _inputLayers.size(); i++) {
                sfDescChunk->_visibleLayerDescs[i]._ignoreMiddle = false;

                sfDescChunk->_visibleLayerDescs[i]._size = { _inputLayers[i]._size.x, _inputLayers[i]._size.y };

                if (_inputLayers[i]._params.find("sfc_ff_numSamples") != _inputLayers[i]._params.end())
                    sfDescChunk->_visibleLayerDescs[i]._numSamples = std::stoi(_inputLayers[i]._params["sfc_ff_numSamples"]);

                if (_inputLayers[i]._params.find("sfc_ff_radius") != _inputLayers[i]._params.end())
                    sfDescChunk->_visibleLayerDescs[i]._radius = std::stoi(_inputLayers[i]._params["sfc_ff_radius"]);

                if (_inputLayers[i]._params.find("sfc_ff_weightAlpha") != _inputLayers[i]._params.end())
                    sfDescChunk->_visibleLayerDescs[i]._weightAlpha = std::stof(_inputLayers[i]._params["sfc_ff_weightAlpha"]);

                if (_inputLayers[i]._params.find("sfc_ff_lambda") != _inputLayers[i]._params.end())
                    sfDescChunk->_visibleLayerDescs[i]._lambda = std::stof(_inputLayers[i]._params["sfc_ff_lambda"]);
            }

            // Recurrent
            /*{
                sfDescChunk->_visibleLayerDescs.back()._ignoreMiddle = true;

                sfDescChunk->_visibleLayerDescs.back()._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };

                if (params.find("sfc_r_numSamples") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._numSamples = std::stoi(params["sfc_r_numSamples"]);

                if (params.find("sfc_r_radius") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._radius = std::stoi(params["sfc_r_radius"]);

                if (params.find("sfc_r_weightAlpha") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._weightAlpha = std::stof(params["sfc_r_weightAlpha"]);

                if (params.find("sfc_r_lambda") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._lambda = std::stof(params["sfc_r_lambda"]);
            }*/
        }
        else {
            sfDescChunk->_visibleLayerDescs.resize(1);

            // Feed forward
            {
                sfDescChunk->_visibleLayerDescs[0]._ignoreMiddle = false;

                sfDescChunk->_visibleLayerDescs[0]._size = { _higherLayers[layerIndex - 1]._size.x, _higherLayers[layerIndex - 1]._size.y };

                if (params.find("sfc_ff_numSamples") != params.end())
                    sfDescChunk->_visibleLayerDescs[0]._numSamples = std::stoi(params["sfc_ff_numSamples"]);

                if (params.find("sfc_ff_radius") != params.end())
                    sfDescChunk->_visibleLayerDescs[0]._radius = std::stoi(params["sfc_ff_radius"]);

                if (params.find("sfc_ff_weightAlpha") != params.end())
                    sfDescChunk->_visibleLayerDescs[0]._weightAlpha = std::stof(params["sfc_ff_weightAlpha"]);

                if (params.find("sfc_ff_lambda") != params.end())
                    sfDescChunk->_visibleLayerDescs[0]._lambda = std::stof(params["sfc_ff_lambda"]);
            }

            // Recurrent
            /*{
                sfDescChunk->_visibleLayerDescs[1]._ignoreMiddle = true;

                sfDescChunk->_visibleLayerDescs[1]._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };

                if (params.find("sfc_r_numSamples") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._numSamples = std::stoi(params["sfc_r_numSamples"]);

                if (params.find("sfc_r_radius") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._radius = std::stoi(params["sfc_r_radius"]);

                if (params.find("sfc_r_weightAlpha") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._weightAlpha = std::stof(params["sfc_r_weightAlpha"]);

                if (params.find("sfc_r_lambda") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._lambda = std::stof(params["sfc_r_lambda"]);
            }*/
        }

        sfDesc = sfDescChunk;

        break;
    }

    case _distance:
    {
        std::shared_ptr<SparseFeaturesDistance::SparseFeaturesDistanceDesc> sfDescDistance = std::make_shared<SparseFeaturesDistance::SparseFeaturesDistanceDesc>();

        sfDescDistance->_cs = _resources->_cs;
        sfDescDistance->_inputType = SparseFeatures::_feedForward;
        sfDescDistance->_hiddenSize = { size.x, size.y };
        sfDescDistance->_rng = _rng;

        if (params.find("sfd_chunkSize") != params.end()) {
            Vec2i chunkSize = ParameterModifier::parseVec2i(params["sfd_chunkSize"]);
            sfDescDistance->_chunkSize = { chunkSize.x, chunkSize.y };
        }

        if (params.find("sfd_gamma") != params.end())
            sfDescDistance->_gamma = std::stof(params["sfd_gamma"]);

        if (params.find("sfd_initWeightRange") != params.end()) {
            Vec2f initWeightRange = ParameterModifier::parseVec2f(params["sfd_initWeightRange"]);
            sfDescDistance->_initWeightRange = { initWeightRange.x, initWeightRange.y };
        }

        if (_resources->_programs.find("distance") == _resources->_programs.end()) {
            _resources->_programs["distance"] = sfDescDistance->_sfdProgram = std::make_shared<ComputeProgram>();

            sfDescDistance->_sfdProgram->loadSparseFeaturesKernel(*_resources->_cs, _distance);
        }
        else
            sfDescDistance->_sfdProgram = _resources->_programs["distance"];

        if (layerIndex == 0) {
            sfDescDistance->_visibleLayerDescs.resize(_inputLayers.size());

            for (int i = 0; i < _inputLayers.size(); i++) {
                sfDescDistance->_visibleLayerDescs[i]._ignoreMiddle = false;

                sfDescDistance->_visibleLayerDescs[i]._size = { _inputLayers[i]._size.x, _inputLayers[i]._size.y };

                if (_inputLayers[i]._params.find("sfd_ff_numSamples") != _inputLayers[i]._params.end())
                    sfDescDistance->_visibleLayerDescs[i]._numSamples = std::stoi(_inputLayers[i]._params["sfd_ff_numSamples"]);

                if (_inputLayers[i]._params.find("sfd_ff_radius") != _inputLayers[i]._params.end())
                    sfDescDistance->_visibleLayerDescs[i]._radius = std::stoi(_inputLayers[i]._params["sfd_ff_radius"]);

                if (_inputLayers[i]._params.find("sfd_ff_weightAlpha") != _inputLayers[i]._params.end())
                    sfDescDistance->_visibleLayerDescs[i]._weightAlpha = std::stof(_inputLayers[i]._params["sfd_ff_weightAlpha"]);

                if (_inputLayers[i]._params.find("sfd_ff_lambda") != _inputLayers[i]._params.end())
                    sfDescDistance->_visibleLayerDescs[i]._lambda = std::stof(_inputLayers[i]._params["sfd_ff_lambda"]);
            }

            // Recurrent
            /*{
                sfDescChunk->_visibleLayerDescs.back()._ignoreMiddle = true;

                sfDescChunk->_visibleLayerDescs.back()._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };

                if (params.find("sfd_r_numSamples") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._numSamples = std::stoi(params["sfd_r_numSamples"]);

                if (params.find("sfd_r_radius") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._radius = std::stoi(params["sfd_r_radius"]);

                if (params.find("sfd_r_weightAlpha") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._weightAlpha = std::stof(params["sfd_r_weightAlpha"]);

                if (params.find("sfd_r_lambda") != params.end())
                    sfDescChunk->_visibleLayerDescs.back()._lambda = std::stof(params["sfd_r_lambda"]);
            }*/
        }
        else {
            sfDescDistance->_visibleLayerDescs.resize(1);

            // Feed forward
            {
                sfDescDistance->_visibleLayerDescs[0]._ignoreMiddle = false;

                sfDescDistance->_visibleLayerDescs[0]._size = { _higherLayers[layerIndex - 1]._size.x, _higherLayers[layerIndex - 1]._size.y };

                if (params.find("sfd_ff_numSamples") != params.end())
                    sfDescDistance->_visibleLayerDescs[0]._numSamples = std::stoi(params["sfd_ff_numSamples"]);

                if (params.find("sfd_ff_radius") != params.end())
                    sfDescDistance->_visibleLayerDescs[0]._radius = std::stoi(params["sfd_ff_radius"]);

                if (params.find("sfd_ff_weightAlpha") != params.end())
                    sfDescDistance->_visibleLayerDescs[0]._weightAlpha = std::stof(params["sfd_ff_weightAlpha"]);

                if (params.find("sfd_ff_lambda") != params.end())
                    sfDescDistance->_visibleLayerDescs[0]._lambda = std::stof(params["sfd_ff_lambda"]);
            }

            // Recurrent
            /*{
                sfDescChunk->_visibleLayerDescs[1]._ignoreMiddle = true;

                sfDescChunk->_visibleLayerDescs[1]._size = { _higherLayers[layerIndex]._size.x, _higherLayers[layerIndex]._size.y };

                if (params.find("sfd_r_numSamples") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._numSamples = std::stoi(params["sfd_r_numSamples"]);

                if (params.find("sfd_r_radius") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._radius = std::stoi(params["sfd_r_radius"]);

                if (params.find("sfd_r_weightAlpha") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._weightAlpha = std::stof(params["sfd_r_weightAlpha"]);

                if (params.find("sfd_r_lambda") != params.end())
                    sfDescChunk->_visibleLayerDescs[1]._lambda = std::stof(params["sfd_r_lambda"]);
            }*/
        }

        sfDesc = sfDescDistance;

        break;
    }
    }

    return sfDesc;
}

void ValueField2D::load(const schemas::ValueField2D* fbValueField2D, ComputeSystem &cs) {
    _size.x = fbValueField2D->_size()->x();
    _size.y = fbValueField2D->_size()->y();

    flatbuffers::uoffset_t numValues = fbValueField2D->_data()->Length();
    _data.resize(numValues);
    for (flatbuffers::uoffset_t i = 0; i < numValues; i++)
        _data[i] = fbValueField2D->_data()->Get(i);
}

flatbuffers::Offset<schemas::ValueField2D> ValueField2D::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::Vec2i size(_size.x, _size.y);
    return schemas::CreateValueField2D(builder,
        builder.CreateVector(_data.data(), _data.size()),
        &size);
}

void ParameterModifier::load(const schemas::ParameterModifier* fbParameterModifier, ComputeSystem &cs) {
    flatbuffers::uoffset_t numValues = fbParameterModifier->_target()->Length();
    for (flatbuffers::uoffset_t i = 0; i < numValues; i++)
    {
        const flatbuffers::String* key = fbParameterModifier->_target()->Get(i)->_key();
        (*_target)[key->c_str()] = fbParameterModifier->_target()->Get(i)->_value()->c_str();
    }
}

flatbuffers::Offset<schemas::ParameterModifier> ParameterModifier::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::Parameter>> parameters;
    for (auto iterator = _target->begin(); iterator != _target->end(); iterator++) {
        parameters.push_back(schemas::CreateParameter(builder,
            builder.CreateString(iterator->first),      //key
            builder.CreateString(iterator->second)));   //value
    }
    return schemas::CreateParameterModifier(builder,
        builder.CreateVector(parameters));
}

void InputLayer::load(const schemas::InputLayer* fbInputLayer, ComputeSystem &cs) {
    _size = Vec2i(fbInputLayer->_size()->x(), fbInputLayer->_size()->y());
    _chunkSize = Vec2i(fbInputLayer->_chunkSize()->x(), fbInputLayer->_chunkSize()->y());
    _isQ = fbInputLayer->_isQ();

    flatbuffers::uoffset_t numValues = fbInputLayer->_params()->Length();
    for (flatbuffers::uoffset_t i = 0; i < numValues; i++)
    {
        const flatbuffers::String* key = fbInputLayer->_params()->Get(i)->_key();
        _params[key->c_str()] = fbInputLayer->_params()->Get(i)->_value()->c_str();
    }
}

flatbuffers::Offset<schemas::InputLayer> InputLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::Vec2i size(_size.x, _size.y);
    schemas::Vec2i chunkSize(_chunkSize.x, _chunkSize.y);

    std::vector<flatbuffers::Offset<schemas::Parameter>> parameters;
    for (auto iterator = _params.begin(); iterator != _params.end(); iterator++) {
        parameters.push_back(schemas::CreateParameter(builder,
            builder.CreateString(iterator->first),      //key
            builder.CreateString(iterator->second)));   //value
    }
    return schemas::CreateInputLayer(builder,
        &size, &chunkSize, _isQ, builder.CreateVector(parameters));
}

void HigherLayer::load(const schemas::HigherLayer* fbHigherLayer, ComputeSystem &cs) {
    _size = Vec2i(fbHigherLayer->_size()->x(), fbHigherLayer->_size()->y());
    
    switch (fbHigherLayer->_type())
    {
    default:
    case schemas::SparseFeaturesTypeEnum__chunk: _type = SparseFeaturesType::_chunk; break;
    case schemas::SparseFeaturesTypeEnum__distance: _type = SparseFeaturesType::_distance; break;
    }

    flatbuffers::uoffset_t numValues = fbHigherLayer->_params()->Length();
    for (flatbuffers::uoffset_t i = 0; i < numValues; i++)
    {
        const flatbuffers::String* key = fbHigherLayer->_params()->Get(i)->_key();
        _params[key->c_str()] = fbHigherLayer->_params()->Get(i)->_value()->c_str();
    }
}

flatbuffers::Offset<schemas::HigherLayer> HigherLayer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::Vec2i size(_size.x, _size.y);

    schemas::SparseFeaturesTypeEnum type;
    switch (_type)
    {
    default:
    case SparseFeaturesType::_chunk:    type = schemas::SparseFeaturesTypeEnum::SparseFeaturesTypeEnum__chunk; break;
    case SparseFeaturesType::_distance:    type = schemas::SparseFeaturesTypeEnum::SparseFeaturesTypeEnum__distance; break;
    }

    std::vector<flatbuffers::Offset<schemas::Parameter>> parameters;
    for (auto iterator = _params.begin(); iterator != _params.end(); iterator++) {
        parameters.push_back(schemas::CreateParameter(builder,
            builder.CreateString(iterator->first),      //key
            builder.CreateString(iterator->second)));   //value
    }
    return schemas::CreateHigherLayer(builder,
        &size, type, builder.CreateVector(parameters));
}

void Architect::load(const schemas::Architect* fbArchitect, ComputeSystem &cs) {
    if (_inputLayers.empty()) {
        _inputLayers.resize(fbArchitect->_inputLayers()->Length());
    }
    else {
        assert(_inputLayers.size() == fbArchitect->_inputLayers()->Length());
    }
    for (flatbuffers::uoffset_t i = 0; i < fbArchitect->_inputLayers()->Length(); i++) {
        _inputLayers[i].load(fbArchitect->_inputLayers()->Get(i), cs);
    }

    if (_higherLayers.empty()) {
        _higherLayers.resize(fbArchitect->_higherLayers()->Length());
    }
    else {
        assert(_higherLayers.size() == fbArchitect->_higherLayers()->Length());
    }
    for (flatbuffers::uoffset_t i = 0; i < fbArchitect->_higherLayers()->Length(); i++) {
        _higherLayers[i].load(fbArchitect->_higherLayers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::Architect> Architect::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::InputLayer>> inputLayers;
    for (InputLayer layer : _inputLayers)
        inputLayers.push_back(layer.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::HigherLayer>> higherLayers;
    for (HigherLayer layer : _higherLayers)
        higherLayers.push_back(layer.save(builder, cs));

    return schemas::CreateArchitect(builder,
        builder.CreateVector(inputLayers),
        builder.CreateVector(higherLayers));
}

void Architect::load(const std::string &fileName) {
    FILE* file = fopen(fileName.c_str(), "rb");
    fseek(file, 0L, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0L, SEEK_SET);
    std::vector<uint8_t> data(length);
    fread(data.data(), sizeof(uint8_t), length, file);
    fclose(file);

    flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);

    bool verified =
        schemas::VerifyArchitectBuffer(verifier) |
        schemas::ArchitectBufferHasIdentifier(data.data());

    if (verified) {
        const schemas::Architect* arch = schemas::GetArchitect(data.data());

        load(arch, *_resources->_cs);
    }

    return; //verified;
}

void Architect::save(const std::string &fileName) {
    flatbuffers::FlatBufferBuilder builder;

    flatbuffers::Offset<schemas::Architect> arch = save(builder, *_resources->_cs);

    // Instruct the builder that this Architect is complete.
    schemas::FinishArchitectBuffer(builder, arch);

    // Get the built buffer and size
    uint8_t *buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier = flatbuffers::Verifier(buf, size);

    bool verified =
        schemas::VerifyArchitectBuffer(verifier) |
        schemas::ArchitectBufferHasIdentifier(buf);

    if (verified) {
        FILE* file = fopen(fileName.c_str(), "wb");
        fwrite(buf, sizeof(uint8_t), size, file);
        fclose(file);
    }

    return; //verified;
}