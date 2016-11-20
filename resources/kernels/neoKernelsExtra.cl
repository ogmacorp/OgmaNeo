// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ----------------------------------------- Sparse Coder -----------------------------------------

void kernel scStimulus(read_only image2d_t visibleStates,
    read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
    int2 visibleSize, float2 hiddenToVisible, int radius, uchar ignoreMiddle)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

    float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;

    float subSum = 0.0f;
    float count = 0.0f;

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            if (ignoreMiddle && dx == 0 && dy == 0)
                continue;

            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

                subSum += visibleState * weight;
                count += 1.0f;
            }
        }

    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum / fmax(1.0f, count)));
}

void kernel scReverse(read_only image2d_t hiddenStates, read_only image2d_t visibleStates,
    write_only image2d_t reconErrors, read_only image3d_t weights,
    int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
    int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);

    float recon = 0.0f;
    float div = 0.0f;

    for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
        for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize)) {
                // Next layer node's receptive field
                int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

                int2 fieldLowerBound = fieldCenter - (int2)(radius);
                int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds

                // Check for containment
                if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {
                    int2 offset = visiblePosition - fieldLowerBound;

                    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

                    int wi = offset.y + offset.x * (radius * 2 + 1);

                    float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                    recon += hiddenState * weight;
                    div += hiddenState;
                }
            }
        }

    float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

    write_imagef(reconErrors, visiblePosition, (float4)(visibleState - recon, 0.0f, 0.0f, 0.0f));
}

void kernel scSolveHidden(read_only image2d_t activations,
    write_only image2d_t hiddenStatesFront,
    int2 hiddenSize, int radius, float activeRatio)
{
    //uint2 seedValue = seed + (uint2)(get_global_id(0) * 51 + 23, get_global_id(1) * 82 + 59) * 24;
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    int2 fieldLowerBound = hiddenPosition - (int2)(radius);

    float activation = read_imagef(activations, defaultSampler, hiddenPosition).x;

    float inhibition = 0.0f;

    float count = 0.0f;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            if (dx == 0 && dy == 0)
                continue;

            int2 otherPosition = hiddenPosition + (int2)(dx, dy);

            if (inBounds0(otherPosition, hiddenSize)) {
                float otherActivation = read_imagef(activations, defaultSampler, otherPosition).x;

                inhibition += otherActivation >= activation ? 1.0f : 0.0f;
                count += 1.0f;
            }
        }

    float state = inhibition <= activeRatio * count ? 1.0f : 0.0f;

    write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state));
}

void kernel scLearnWeights(read_only image2d_t hiddenStates, read_only image2d_t hiddenStatesPrev,
    read_only image2d_t reconErrors,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, float activeRatio, float weightAlpha)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;
    float hiddenStatePrev = read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).x;

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float reconError = read_imagef(reconErrors, defaultSampler, visiblePosition).x;

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weightPrev + weightAlpha * hiddenState * reconError));
            }
        }
}

void kernel scLearnThresholds(read_only image2d_t hiddenStates, read_only image2d_t hiddenThresholdsBack, write_only image2d_t hiddenThresholdsFront, float thresholdAlpha, float activeRatio) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

    float hiddenThresholdPrev = read_imagef(hiddenThresholdsBack, defaultSampler, hiddenPosition).x;

    write_imagef(hiddenThresholdsFront, hiddenPosition, (float4)(hiddenThresholdPrev + thresholdAlpha * (activeRatio - hiddenState), 0.0f, 0.0f, 0.0f));
}

void kernel scDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront, float lambda) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float input = read_imagef(inputs, defaultSampler, position).x;

    float2 outputPrev = read_imagef(outputsBack, defaultSampler, position).xy;

    write_imagef(outputsFront, position, (float4)(input, lambda * outputPrev.y + (1.0f - lambda) * input, 0.0f, 0.0f));
}

void kernel scReconstruct(read_only image2d_t hiddenStates,
    write_only image2d_t reconstruction, read_only image3d_t weights,
    int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
    int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = (int2)(visiblePosition.x * visibleToHidden.x + 0.5f, visiblePosition.y * visibleToHidden.y + 0.5f);

    float recon = 0.0f;
    float div = 0.0f;

    for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
        for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize)) {
                // Next layer node's receptive field
                int2 fieldCenter = (int2)(hiddenPosition.x * hiddenToVisible.x + 0.5f, hiddenPosition.y * hiddenToVisible.y + 0.5f);

                int2 fieldLowerBound = fieldCenter - (int2)(radius);
                int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds

                // Check for containment
                if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {
                    int2 offset = visiblePosition - fieldLowerBound;

                    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

                    int wi = offset.y + offset.x * (radius * 2 + 1);

                    float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                    recon += hiddenState * weight;
                    div += hiddenState;
                }
            }
        }

    write_imagef(reconstruction, visiblePosition, (float4)(recon, 0.0f, 0.0f, 0.0f));
}

// ----------------------------------------- Preprocessing -----------------------------------------

void kernel whiten(read_only image2d_t input, write_only image2d_t result, int2 imageSize, int kernelRadius, float intensity) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float4 currentColor = read_imagef(input, defaultSampler, position);

    float4 center = currentColor;

    float count = 0.0f;

    for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
        for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
            if (dx == 0 && dy == 0)
                continue;

            int2 otherPosition = position + (int2)(dx, dy);

            if (inBounds0(otherPosition, imageSize)) {
                float4 otherColor = read_imagef(input, defaultSampler, otherPosition);

                center += otherColor;

                count += 1.0f;
            }
        }

    center /= count + 1.0f;

    float4 centeredCurrentColor = currentColor - center;

    float4 covariances = (float4)(0.0f);

    for (int dx = -kernelRadius; dx <= kernelRadius; dx++)
        for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
            if (dx == 0 && dy == 0)
                continue;

            int2 otherPosition = position + (int2)(dx, dy);

            if (inBounds0(otherPosition, imageSize)) {
                float4 otherColor = read_imagef(input, defaultSampler, otherPosition);

                float4 centeredOtherColor = otherColor - center;

                covariances += centeredOtherColor * centeredCurrentColor;
            }
        }

    covariances /= fmax(1.0f, count);

    float4 centeredCurrentColorSigns = (float4)(centeredCurrentColor.x > 0.0f ? 1.0f : -1.0f,
        centeredCurrentColor.y > 0.0f ? 1.0f : -1.0f,
        centeredCurrentColor.z > 0.0f ? 1.0f : -1.0f,
        centeredCurrentColor.w > 0.0f ? 1.0f : -1.0f);

    // Modify color
    float4 whitenedColor = fmin(1.0f, fmax(-1.0f, (centeredCurrentColor > 0.0f ? (float4)(1.0f) : (float4)(-1.0f)) * (1.0f - exp(-fabs(intensity * covariances)))));

    write_imagef(result, position, whitenedColor);
}
