// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "system/SharedLib.h"
#include "system/ComputeSystem.h"
#include "system/ComputeProgram.h"
#include "Helpers.h"

namespace ogmaneo {
    /*!
    \brief Sparse coder
    Learns a spatial-only sparse code
    */
    class OGMA_API SparseCoder {
    public:
        /*!
        \brief Visible layer descriptor
        */
        struct VisibleLayerDesc {
            /*!
            \brief Size of layer
            */
            cl_int2 _size;

            /*!
            \brief Radius onto input
            */
            cl_int _radius;

            /*!
            \brief Whether or not the middle (center) input should be ignored (self in recurrent schemes)
            */
            unsigned char _ignoreMiddle;

            /*!
            \brief Learning rate
            */
            cl_float _weightAlpha;

            /*!
            \brief Initialize defaults
            */
            VisibleLayerDesc()
                : _size({ 8, 8 }), _radius(6), _ignoreMiddle(false),
                _weightAlpha(0.001f)
            {}
        };

        /*!
        \brief Visible layer
        */
        struct VisibleLayer {
            /*!
            \brief Possibly manipulated input
            */
            DoubleBuffer2D _derivedInput;

            /*!
            \brief Temporary buffer for reconstruction error
            */
            cl::Image2D _reconError;

            //!@{
            /*!
            \brief Weights
            */
            DoubleBuffer3D _weights; // Encoding weights (creates spatio-temporal sparse code)
            //!@}

            //!@{
            /*!
            \brief Transformations
            */
            cl_float2 _hiddenToVisible;
            cl_float2 _visibleToHidden;

            cl_int2 _reverseRadii;
            //!@}
        };

    private:
        //!@{
        /*!
        \brief Hidden states, thresholds (similar to biases)
        */
        DoubleBuffer2D _hiddenStates;
        DoubleBuffer2D _hiddenThresholds;
        //!@}

        /*!
        \brief Hidden size
        */
        cl_int2 _hiddenSize;

        /*!
        \brief Inhibition radius
        */
        int _inhibitionRadius;

        /*!
        \brief Hidden stimulus summation temporary buffer
        */
        DoubleBuffer2D _hiddenStimulusSummationTemp;

        //!@{
        /*!
        \brief Layers and descs
        */
        std::vector<VisibleLayerDesc> _visibleLayerDescs;
        std::vector<VisibleLayer> _visibleLayers;
        //!@}

        //!@{
        /*!
        \brief Kernels
        */
        cl::Kernel _stimulusKernel;
        cl::Kernel _reverseKernel;
        cl::Kernel _reconstructKernel;
        cl::Kernel _solveHiddenKernel;
        cl::Kernel _learnWeightsKernel;
        cl::Kernel _learnThresholdsKernel;
        cl::Kernel _deriveInputsKernel;
        //!@}

    public:
        /*!
        \brief Create a comparison sparse coder with random initialization.
        Requires the ComputeSystem, ComputeProgram with the OgmaNeo kernels, and initialization information.
        \param cs is the ComputeSystem.
        \param program is the ComputeProgram associated with the ComputeSystem and loaded with the main kernel code.
        \param visibleLayerDescs descriptors for all input layers.
        \param hiddenSize hidden (output) size (2D).
        \param inhibitionRadius inhibitory radius.
        \param initWeightRange are the minimum and maximum range values for weight initialization.
        \param initThresholdRange are the minimum and maximum range values for threshold initialization.
        \param rng a random number generator.
        */
        void createRandom(ComputeSystem &cs, ComputeProgram &program,
            const std::vector<VisibleLayerDesc> &visibleLayerDescs, cl_int2 hiddenSize,
            int inhibitionRadius, cl_float2 initWeightRange, cl_float2 initThresholdRange,
            std::mt19937 &rng);

        /*!
        \brief Activate predictor
        \param cs is the ComputeSystem.
        \param visibleStates input layer states.
        \param inputTraceDecay decay of input averaging trace.
        \param activeRatio % active units.
        \param rng a random number generator.
        */
        void activate(ComputeSystem &cs, const std::vector<cl::Image2D> &visibleStates, float inputTraceDecay, float activeRatio, std::mt19937 &rng);
        
        /*!
        \brief End a simulation step
        \param cs is the ComputeSystem.
        */
        void stepEnd(ComputeSystem &cs);

        /*!
        \brief Learning
        \param cs is the ComputeSystem.
        \param thresholdAlpha threshold learning rate.
        \param activeRatio % active units.
        */
        void learn(ComputeSystem &cs,
            float thresholdAlpha, float activeRatio);

        /*!
        \brief Reconstruct an SDR
        */
        void reconstruct(ComputeSystem &cs, const cl::Image2D &hiddenStates, std::vector<cl::Image2D> &reconstructions);

        /*!
        \brief Get number of visible layers
        */
        size_t getNumVisibleLayers() const {
            return _visibleLayers.size();
        }

        /*!
        \brief Get access to visible layer
        */
        const VisibleLayer &getVisibleLayer(int index) const {
            return _visibleLayers[index];
        }

        /*!
        \brief Get access to visible layer
        */
        const VisibleLayerDesc &getVisibleLayerDesc(int index) const {
            return _visibleLayerDescs[index];
        }

        /*!
        \brief Get hidden size
        */
        cl_int2 getHiddenSize() const {
            return _hiddenSize;
        }

        /*!
        \brief Get hidden states
        */
        const DoubleBuffer2D &getHiddenStates() const {
            return _hiddenStates;
        }

        /*!
        \brief Get hidden biases
        */
        const DoubleBuffer2D &getHiddenThresholds() const {
            return _hiddenThresholds;
        }

        /*!
        \brief Clear the working memory
        \param cs is the ComputeSystem.
        */
        void clearMemory(ComputeSystem &cs);
    };
}