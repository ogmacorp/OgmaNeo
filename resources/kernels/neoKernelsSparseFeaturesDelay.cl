// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ----------------------------------------- Delay Encoder -----------------------------------------

void kernel sfdStimulus(read_only image2d_t visibleStates,
    read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
    int2 visibleSize, float2 hiddenToVisible, int radius, uchar ignoreMiddle)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;

    float subSum = 0.0f;
	
	float stateSum = 0.0f;

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

                subSum += weight * visibleState;
				stateSum += visibleState * visibleState;
            }
        }

    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum, 0.0f, 0.0f, 0.0f));
}

void kernel sfdActivate(read_only image2d_t stimuli, read_only image2d_t hiddenStates, read_only image2d_t biases,
    read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    float stimulus = read_imagef(stimuli, defaultSampler, hiddenPosition).x;

    float activationPrev = read_imagef(hiddenActivationsBack, defaultSampler, hiddenPosition).x;

    float statePrev = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

    float bias = read_imagef(biases, defaultSampler, hiddenPosition).x;

    float activation = fmax(0.0f, activationPrev * (1.0f - statePrev) + stimulus + bias);

    write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(activation, 0.0f, 0.0f, 0.0f));
}

void kernel sfdInhibit(read_only image2d_t activations,
    write_only image2d_t hiddenStatesFront,
    int2 hiddenSize, int radius, float activeRatio)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

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

    float state = inhibition < activeRatio * count ? 1.0f : 0.0f;

    write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state, 0.0f, 0.0f, 0.0f));
}

void kernel sfdLearnWeights(read_only image2d_t hiddenStates, read_only image2d_t hiddenStatesPrev,
    read_only image2d_t visibleStates, read_only image2d_t visibleStatesPrev,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, float activeRatio, float weightAlpha, float lambda, float gamma)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    float2 hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).xy;
    float2 hiddenStatePrev = read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).xy;

	float weightSum = 0.0f;
	
	for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                weightSum += weightPrev * weightPrev;
            }
        }
		
	float scale = 1.0f / fmax(0.0001f, weightSum);
	
    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float3 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xyz;

                float2 visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).xy;
                float2 visibleStatePrev = read_imagef(visibleStatesPrev, defaultSampler, visiblePosition).xy;

                float traceShort = weightPrev.y * lambda + (1.0f - lambda) * hiddenState.x * visibleState.y;
                float traceLong = weightPrev.z * gamma + (1.0f - gamma) * hiddenState.x * visibleState.y;

                float learn = traceLong - traceShort;

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weightPrev.x * scale + weightAlpha * learn, traceShort, traceLong, 0.0f));
            }
        }
}

void kernel sfdLearnBiases(read_only image2d_t stimuli, read_only image2d_t hiddenStates, read_only image2d_t hiddenBiasesBack, write_only image2d_t hiddenBiasesFront, float activeRatio, float biasAlpha) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

    float stimulus = read_imagef(stimuli, defaultSampler, hiddenPosition).x;
    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

    float hiddenBiasPrev = read_imagef(hiddenBiasesBack, defaultSampler, hiddenPosition).x;

    write_imagef(hiddenBiasesFront, hiddenPosition, (float4)(hiddenBiasPrev + biasAlpha * (-stimulus - hiddenBiasPrev), 0.0f, 0.0f, 0.0f));
}

void kernel sfdDeriveInputs(read_only image2d_t inputs, write_only image2d_t outputsFront) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float input = read_imagef(inputs, defaultSampler, position).x;

    write_imagef(outputsFront, position, (float4)(input, 0.0f, 0.0f, 0.0f));
}