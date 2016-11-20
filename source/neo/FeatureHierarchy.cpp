// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "FeatureHierarchy.h"
#include "SparseFeaturesChunk.h"
#include "SparseFeaturesDelay.h"
#include "SparseFeaturesSTDP.h"

using namespace ogmaneo;

void FeatureHierarchy::createRandom(ComputeSystem &cs, ComputeProgram &fhProgram,
    const std::vector<LayerDesc> &layerDescs,
    std::mt19937 &rng)
{
    _layerDescs = layerDescs;

    _layers.resize(_layerDescs.size());

    for (int l = 0; l < _layers.size(); l++) {
        _layers[l]._sf = _layerDescs[l]._sfDesc->sparseFeaturesFactory();

        // Create temporal pooling buffer
        _layers[l]._tpBuffer = createDoubleBuffer2D(cs, _layers[l]._sf->getHiddenSize(), CL_R, CL_FLOAT);

        // Prediction error
        _layers[l]._predErrors = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), _layers[l]._sf->getHiddenSize().x, _layers[l]._sf->getHiddenSize().y);
    }

    // Kernels
    _fhPoolKernel = cl::Kernel(fhProgram.getProgram(), "fhPool");
    _fhPredErrorKernel = cl::Kernel(fhProgram.getProgram(), "fhPredError");
}

void FeatureHierarchy::simStep(ComputeSystem &cs, const std::vector<cl::Image2D> &inputs, const std::vector<cl::Image2D> &predictionsPrev, std::mt19937 &rng, bool learn) {
    // Clear summation buffers if reset previously
    for (int l = 0; l < _layers.size(); l++) {
        if (_layers[l]._tpNextReset)
            // Clear summation buffer
            cs.getQueue().enqueueFillImage(_layers[l]._tpBuffer[_back], cl_float4{ 0.0f, 0.0f, 0.0f, 0.0f }, { 0, 0, 0 }, { static_cast<cl::size_type>(_layers[l]._sf->getHiddenSize().x), static_cast<cl::size_type>(_layers[l]._sf->getHiddenSize().y), 1 });
    }

    // Activate
    bool prevClockReset = true;

    for (int l = 0; l < _layers.size(); l++) {
        // Add input to pool
        if (prevClockReset) {
            _layers[l]._clock++;

            // Gather inputs for layer
            std::vector<cl::Image2D> visibleStates;

            if (l == 0) {
                std::vector<cl::Image2D> inputsUse = inputs;

                if (_layerDescs.front()._sfDesc->_inputType == SparseFeatures::_feedForwardRecurrent)
                    inputsUse.push_back(_layers.front()._sf->getHiddenContext());

                visibleStates = inputsUse;
            }
            else
                visibleStates = _layerDescs[l]._sfDesc->_inputType == SparseFeatures::_feedForwardRecurrent ? std::vector<cl::Image2D>{ _layers[l - 1]._tpBuffer[_back], _layers[l]._sf->getHiddenContext() } : std::vector<cl::Image2D>{ _layers[l - 1]._tpBuffer[_back] };

            // Update layer
            _layers[l]._sf->activate(cs, visibleStates, predictionsPrev[l], rng);

            if (learn)
                _layers[l]._sf->learn(cs, rng);

            _layers[l]._sf->stepEnd(cs);

            // Prediction error
            {
                int argIndex = 0;

                _fhPredErrorKernel.setArg(argIndex++, _layers[l]._sf->getHiddenStates()[_back]);
                _fhPredErrorKernel.setArg(argIndex++, predictionsPrev[l]);
                _fhPredErrorKernel.setArg(argIndex++, _layers[l]._predErrors);

                cs.getQueue().enqueueNDRangeKernel(_fhPredErrorKernel, cl::NullRange, cl::NDRange(_layers[l]._sf->getHiddenSize().x, _layers[l]._sf->getHiddenSize().y));
            }

            // Add state to average
            {
                int argIndex = 0;

                _fhPoolKernel.setArg(argIndex++, _layers[l]._predErrors);
                _fhPoolKernel.setArg(argIndex++, _layers[l]._tpBuffer[_back]);
                _fhPoolKernel.setArg(argIndex++, _layers[l]._tpBuffer[_front]);
                _fhPoolKernel.setArg(argIndex++, 1.0f / std::max(1, _layerDescs[l]._poolSteps));

                cs.getQueue().enqueueNDRangeKernel(_fhPoolKernel, cl::NullRange, cl::NDRange(_layers[l]._sf->getHiddenSize().x, _layers[l]._sf->getHiddenSize().y));

                std::swap(_layers[l]._tpBuffer[_front], _layers[l]._tpBuffer[_back]);
            }
        }

        _layers[l]._tpReset = prevClockReset;

        if (_layers[l]._clock >= _layerDescs[l]._poolSteps) {
            _layers[l]._clock = 0;

            prevClockReset = true;
        }
        else
            prevClockReset = false;

        _layers[l]._tpNextReset = prevClockReset;
    }
}

void FeatureHierarchy::clearMemory(ComputeSystem &cs) {
    for (int l = 0; l < _layers.size(); l++)
        _layers[l]._sf->clearMemory(cs);
}

void FeatureHierarchy::LayerDesc::load(const schemas::FeatureHierarchyLayerDesc* fbFeatureHierarchyLayerDesc, ComputeSystem &cs) {
    _sfDesc->load(fbFeatureHierarchyLayerDesc->_sfDesc(), cs);
    _poolSteps = fbFeatureHierarchyLayerDesc->_poolSteps();
}

flatbuffers::Offset<schemas::FeatureHierarchyLayerDesc> FeatureHierarchy::LayerDesc::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    return schemas::CreateFeatureHierarchyLayerDesc(builder,
        _sfDesc->save(builder, cs), _poolSteps);
}

void FeatureHierarchy::Layer::load(const schemas::FeatureHierarchyLayer* fbFeatureHierarchyLayer, ComputeSystem &cs) {
    schemas::SparseFeatures* fbSparseFeatures =
        (schemas::SparseFeatures*)(fbFeatureHierarchyLayer->_sf());
    _sf->load(fbSparseFeatures, cs);

    _clock = fbFeatureHierarchyLayer->_clock();
    ogmaneo::load(_tpBuffer, fbFeatureHierarchyLayer->_tpBuffer(), cs);
    ogmaneo::load(_predErrors, fbFeatureHierarchyLayer->_predErrors(), cs);
    _tpReset = fbFeatureHierarchyLayer->_tpReset();
    _tpNextReset = fbFeatureHierarchyLayer->_tpNextReset();
}

flatbuffers::Offset<schemas::FeatureHierarchyLayer> FeatureHierarchy::Layer::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    schemas::SparseFeaturesType type;
    switch (_sf->_type) {
    default:
    case SparseFeaturesType::_chunk:    type = schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesChunk; break;
    case SparseFeaturesType::_delay:    type = schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesDelay; break;
    case SparseFeaturesType::_stdp:  type = schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesSTDP; break;
    }
    return schemas::CreateFeatureHierarchyLayer(builder,
        type, _sf->save(builder, cs).Union(),
        _clock,
        ogmaneo::save(_tpBuffer, builder, cs),
        ogmaneo::save(_predErrors, builder, cs),
        _tpReset, _tpNextReset);
}

void FeatureHierarchy::load(const schemas::FeatureHierarchy* fbFeatureHierarchy, ComputeSystem &cs) {
    if (!_layers.empty()) {
        assert(_layerDescs.size() == fbFeatureHierarchy->_layerDescs()->Length());
        assert(_layers.size() == fbFeatureHierarchy->_layers()->Length());
    }
    else {
        _layerDescs.reserve(fbFeatureHierarchy->_layerDescs()->Length());
        _layers.reserve(fbFeatureHierarchy->_layers()->Length());
    }

    for (flatbuffers::uoffset_t i = 0; i < fbFeatureHierarchy->_layerDescs()->Length(); i++) {
        _layerDescs[i].load(fbFeatureHierarchy->_layerDescs()->Get(i), cs);
    }

    for (flatbuffers::uoffset_t i = 0; i < fbFeatureHierarchy->_layers()->Length(); i++) {
        _layers[i].load(fbFeatureHierarchy->_layers()->Get(i), cs);
    }
}

flatbuffers::Offset<schemas::FeatureHierarchy> FeatureHierarchy::save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs) {
    std::vector<flatbuffers::Offset<schemas::FeatureHierarchyLayerDesc>> layerDescs;
    for (LayerDesc layerDesc : _layerDescs)
        layerDescs.push_back(layerDesc.save(builder, cs));

    std::vector<flatbuffers::Offset<schemas::FeatureHierarchyLayer>> layers;
    for (Layer layer : _layers)
        layers.push_back(layer.save(builder, cs));

    return schemas::CreateFeatureHierarchy(builder,
        builder.CreateVector(layerDescs), builder.CreateVector(layers));
}