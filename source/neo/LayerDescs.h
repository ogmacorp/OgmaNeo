// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

namespace ogmaneo {
    /*!
    \brief Layer descriptor for other classes and language bindings to use
    */
    struct LayerDescs {
        //!@{
        /*!
        \brief Layer size
        */
        int _width, _height;
        //!@}

        //!@{
        /*!
        \brief Feature hierarchy parameters
        */
        int _feedForwardRadius, _recurrentRadius, _inhibitionRadius;
        float _spFeedForwardWeightAlpha;
        float _spRecurrentWeightAlpha;
        float _spBiasAlpha;
        float _spActiveRatio;
        //!@}

        //!@{
        /*!
        \brief Predictor layer parameters
        */
        int _predRadius;
        float _predAlpha;
        float _predBeta;
        //!@}

        //!@{
        /*!
        \brief Agent layer parameters
        */
        int _qRadius;
        float _qAlpha;
        float _qGamma;
        float _qLambda;
        float _epsilon;
        //!@}

        /*!
        \brief Initialize defaults
        */
        LayerDescs()
            : _width(8), _height(8),
            // Feature hierarchy parameters
            _feedForwardRadius(6), _recurrentRadius(6), _inhibitionRadius(5),
            _spFeedForwardWeightAlpha(0.1f), _spRecurrentWeightAlpha(0.1f),
            _spBiasAlpha(0.001f),
            _spActiveRatio(0.04f),
            // Predictor parameters
            _predRadius(8), _predAlpha(0.06f), _predBeta(0.1f),
            // Agent Q parameters
            _qRadius(10), _qAlpha(0.01f), _qGamma(0.994f), _qLambda(0.99f), _epsilon(0.04f)
        {}

        /*!
        \brief Initialize defaults
        */
        LayerDescs(int width, int height)
            : _width(width), _height(height),
            // Feature hierarchy parameters
            _feedForwardRadius(6), _recurrentRadius(6), _inhibitionRadius(5),
            _spFeedForwardWeightAlpha(0.1f), _spRecurrentWeightAlpha(0.1f),
            _spBiasAlpha(0.001f),
            _spActiveRatio(0.04f),
            // Predictor parameters
            _predRadius(8), _predAlpha(0.06f), _predBeta(0.1f),
            // Agent Q parameters
            _qRadius(10), _qAlpha(0.01f), _qGamma(0.99f), _qLambda(0.98f), _epsilon(0.04f)
        {}
    };
}