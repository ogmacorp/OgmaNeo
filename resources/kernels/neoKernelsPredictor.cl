// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ------------------------------------------- Predictor Layer -------------------------------------------

void kernel plDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float input = read_imagef(inputs, defaultSampler, position).x;

	float outputPrev = read_imagef(outputsBack, defaultSampler, position).x;

    write_imagef(outputsFront, position, (float4)(input, fmax(0.0f, input - outputPrev), 0.0f, 0.0f));
}

void kernel plStimulus(read_only image2d_t visibleStates,
    read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront,
    read_only image3d_t weights,
    int2 visibleSize, float2 hiddenToVisible, int radius)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;

    float subSum = 0.0f;
	float stateSum = 0.0f;

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

                subSum += visibleState * weight;
				stateSum += visibleState;
            }
        }

    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum / fmax(0.0001f, stateSum), 0.0f, 0.0f, 0.0f));
}

void kernel plLearnPredWeights(read_only image2d_t visibleStatesPrev,
    read_only image2d_t targets, read_only image2d_t hiddenStatesPrev,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, float alpha)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    float error = read_imagef(targets, defaultSampler, hiddenPosition).x - read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).x;

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float2 visibleStatePrev = read_imagef(visibleStatesPrev, defaultSampler, visiblePosition).xy;

                float weight = weightPrev + alpha * error * visibleStatePrev.x * visibleStatePrev.y;

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
            }
        }
}

void kernel plThreshold(read_only image2d_t stimuli, write_only image2d_t thresholded) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    float stimulus = read_imagef(stimuli, defaultSampler, hiddenPosition).x;

    write_imagef(thresholded, hiddenPosition, (float4)(stimulus > 0.5f ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));//fmax(0.0f, fabs(stimulus) - 0.5f) * (stimulus > 0.0f ? 1.0f : -1.0f), 0.0f, 0.0f, 0.0f));
}