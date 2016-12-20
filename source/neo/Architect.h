// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "Predictor.h"
#include "AgentSwarm.h"
#include "schemas/Architect_generated.h"

#include <unordered_map>
#include <sstream>

namespace ogmaneo {
    /*!
    \brief Simple 2D integer vector
    */
    class OGMA_API Vec2i {
    public:
        int x, y;

        Vec2i()
            : x(16), y(16)
        {}

        Vec2i(int X, int Y)
            : x(X), y(Y)
        {}
    };

    /*!
    \brief Simple 2D float vector
    */
    class OGMA_API Vec2f {
    public:
        float x, y;

        Vec2f()
            : x(16.0f), y(16.0f)
        {}

        Vec2f(float X, float Y)
            : x(X), y(Y)
        {}

        Vec2f(int X, int Y)
            : x(static_cast<float>(X)), y(static_cast<float>(Y))
        {}
    };

    /*!
    \brief Shared resources
    */
    class OGMA_API Resources {
    private:
        std::shared_ptr<ComputeSystem> _cs;
        std::unordered_map<std::string, std::shared_ptr<ComputeProgram>> _programs;

    public:
        Resources()
        {}

        Resources(ComputeSystem::DeviceType type, int platformIndex = -1, int deviceIndex = -1) {
            create(type, platformIndex, deviceIndex);
        }

        void create(ComputeSystem::DeviceType type, int platformIndex = -1, int deviceIndex = -1) {
            _cs = std::make_shared<ComputeSystem>();
            _cs->create(type, platformIndex, deviceIndex);
        }

        const std::shared_ptr<ComputeSystem> &getComputeSystem() const {
            return _cs;
        }

        const std::unordered_map<std::string, std::shared_ptr<ComputeProgram>> &getPrograms() const {
            return _programs;
        }

        friend class Architect;
        friend class Hierarchy;
        friend class Agent;
    };

    /*!
    \brief Describe a 2D field of values
    Typically used for input values to a hierarchy.
    */
    class OGMA_API ValueField2D {
    private:
        std::vector<float> _data;
        Vec2i _size;

    public:
        ValueField2D()
        {}

        ValueField2D(const Vec2i &size, float defVal = 0.0f) {
            create(size, defVal);
        }

        void create(const Vec2i &size, float defVal = 0.0f) {
            _size = size;

            _data.clear();
            _data.assign(size.x * size.y, defVal);
        }

        float getValue(const Vec2i &pos) const {
            return _data[pos.x + pos.y * _size.x];
        }

        void setValue(const Vec2i &pos, float value) {
            _data[pos.x + pos.y * _size.x] = value;
        }

        const Vec2i getSize() const {
            return _size;
        }

        std::vector<float> &getData() {
            return _data;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::ValueField2D* fbValueField2D, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::ValueField2D> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };

    /*!
    \brief Parameter modified interface
    */
    class OGMA_API ParameterModifier {
    public:
        static const std::string _boolTrue;
        static const std::string _boolFalse;

    private:
        std::unordered_map<std::string, std::string>* _target;

    public:
        ParameterModifier &setValue(const std::string &name, const std::string &value) {
            (*_target)[name] = value;

            return *this;
        }

        ParameterModifier &setValueBool(const std::string &name, bool value) {
            (*_target)[name] = value ? _boolTrue : _boolFalse;

            return *this;
        }

        ParameterModifier &setValue(const std::string &name, float value) {
            (*_target)[name] = std::to_string(value);

            return *this;
        }

        ParameterModifier &setValue(const std::string &name, const Vec2i &size) {
            (*_target)[name] = "(" + std::to_string(size.x) + ", " + std::to_string(size.y) + ")";

            return *this;
        }

        ParameterModifier &setValue(const std::string &name, const Vec2f &size) {
            (*_target)[name] = "(" + std::to_string(size.x) + ", " + std::to_string(size.y) + ")";

            return *this;
        }

        ParameterModifier &setValues(const std::vector<std::pair<std::string, std::string>> &namesValues) {
            for (int i = 0; i < namesValues.size(); i++)
                setValue(std::get<0>(namesValues[i]), std::get<1>(namesValues[i]));

            return *this;
        }

        static Vec2i parseVec2i(const std::string &s) {
            std::istringstream is(s.substr(1, s.size() - 2)); // Remove ()

            std::string xs;
            std::getline(is, xs, ',');

            std::string ys;
            std::getline(is, ys);

            return Vec2i(std::stoi(xs), std::stoi(ys));
        }

        static Vec2f parseVec2f(const std::string &s) {
            std::istringstream is(s.substr(1, s.size() - 2)); // Remove ()

            std::string xs;
            std::getline(is, xs, ',');

            std::string ys;
            std::getline(is, ys);

            return Vec2f(std::stof(xs), std::stof(ys));
        }

        static bool parseBool(const std::string &s) {
            return s == _boolTrue;
        }

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::ParameterModifier* fbParameterModifier, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::ParameterModifier> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}

        friend class Architect;
    };

    struct InputLayer {
        Vec2i _size;

        std::unordered_map<std::string, std::string> _params;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::InputLayer* fbInputLayer, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::InputLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };

    struct ActionLayer {
        Vec2i _size;

        Vec2i _tileSize;

        std::unordered_map<std::string, std::string> _params;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::ActionLayer* fbActionLayer, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::ActionLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };

    struct HigherLayer {
        Vec2i _size;

        SparseFeaturesType _type;

        std::unordered_map<std::string, std::string> _params;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::HigherLayer* fbHigherLayer, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::HigherLayer> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);
        //!@}
    };

    /*!
    \brief Hierarchy architect
    Used to create hierarchies with a simple interface. Generates an agent or a hierarchy based on the specifications provided.
    */
    class OGMA_API Architect {
    private:
        std::vector<InputLayer> _inputLayers;
        std::vector<ActionLayer> _actionLayers;
        std::vector<HigherLayer> _higherLayers;

        std::mt19937 _rng;

        std::shared_ptr<SparseFeatures::SparseFeaturesDesc> sfDescFromName(
            int layerIndex, SparseFeaturesType type, const Vec2i &size,
            SparseFeatures::InputType inputType, std::unordered_map<std::string, std::string> &params);

        std::shared_ptr<Resources> _resources;

        //!@{
        /*!
        \brief Serialization
        */
        void load(const ogmaneo::schemas::Architect* fbArchitect, ComputeSystem &cs);
        flatbuffers::Offset<ogmaneo::schemas::Architect> save(flatbuffers::FlatBufferBuilder &builder, ComputeSystem &cs);

    public:
        void initialize(unsigned int seed, const std::shared_ptr<Resources> &resources);

        ParameterModifier addInputLayer(const Vec2i &size);
        ParameterModifier addActionLayer(const Vec2i &size, const Vec2i &tileSize);

        ParameterModifier addHigherLayer(const Vec2i &size, SparseFeaturesType type);

        std::shared_ptr<class Hierarchy> generateHierarchy() {
            std::unordered_map<std::string, std::string> emptyHierarchy;
            return generateHierarchy(emptyHierarchy);
        }

        std::shared_ptr<class Agent> generateAgent() {
            std::unordered_map<std::string, std::string> emptyAgentHierarchy;
            return generateAgent(emptyAgentHierarchy);
        }

        std::shared_ptr<class Hierarchy> generateHierarchy(std::unordered_map<std::string, std::string> &additionalParams);
        std::shared_ptr<class Agent> generateAgent(std::unordered_map<std::string, std::string> &additionalParams);

        //!@{
        /*!
        \brief Serialization
        */
        void load(const std::string &fileName);
        void save(const std::string &fileName);
        //!@}
    };
}