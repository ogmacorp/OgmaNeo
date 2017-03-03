// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ------------------------------------------- Predictor Layer -------------------------------------------

void kernel plDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront, float gamma) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float input = read_imagef(inputs, defaultSampler, position).x;
	
	//float2 outputPrev = read_imagef(outputsBack, defaultSampler, position).xy;

    write_imagef(outputsFront, position, (float4)(input, 0.0f, 0.0f, 0.0f));
    //write_imagef(outputsFront, position, (float4)(input, fmax(0.0f, input - outputPrev.x), 0.0f, 0.0f));
}

void kernel plStimulus(read_only image2d_t visibleStates,
    read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront,
    read_only image3d_t weights,
    int2 visibleSize, float2 hiddenToVisible, int radius, int2 chunkSize)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
	
    int2 visiblePositionCenter = project(chunkPosition, hiddenToVisible);

    float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;

    float subSum = 0.0f;

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
            }
        }

    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum, 0.0f, 0.0f, 0.0f));
}

void kernel plInhibitBinary(read_only image2d_t activations,
	write_only image2d_t hiddenStatesFront,
	int2 hiddenSize, int2 chunkSize)
{
	int2 chunkPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 hiddenStartPosition = chunkPosition * chunkSize;

	float maxValue = -99999.0f;
	int2 maxDelta = (int2)(0);
	
	for (int dx = 0; dx < chunkSize.x; dx++)
		for (int dy = 0; dy < chunkSize.y; dy++) {
			int2 hiddenPosition = hiddenStartPosition + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
                float activation = read_imagef(activations, defaultSampler, hiddenPosition).x;

				if (activation > maxValue) {
					maxValue = activation;
					
					maxDelta = (int2)(dx, dy);
				}
			}
		}
			
    for (int dx = 0; dx < chunkSize.x; dx++)
		for (int dy = 0; dy < chunkSize.y; dy++) {
			int2 hiddenPosition = hiddenStartPosition + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				float activation = read_imagef(activations, defaultSampler, hiddenPosition).x;
				
				float hiddenState = (dx == maxDelta.x && dy == maxDelta.y) ? 1.0f : 0.0f;
                
				write_imagef(hiddenStatesFront, hiddenPosition, (float4)(hiddenState, activation, 0.0f, 0.0f));
			}
		}
}

void kernel plLearnPredWeights(read_only image2d_t visibleStatesPrev,
    read_only image2d_t targets, read_only image2d_t hiddenStatesPrev,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, int2 chunkSize, float weightAlpha)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
	
    int2 visiblePositionCenter = project(chunkPosition, hiddenToVisible);
	
    float error = read_imagef(targets, defaultSampler, hiddenPosition).x - read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).x;

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float visibleStatePrev = read_imagef(visibleStatesPrev, defaultSampler, visiblePosition).x;

				float weight = weightPrev + weightAlpha * error * visibleStatePrev;

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
            }
        }
}

void kernel plLearnPredWeightsBinary(read_only image2d_t visibleStatesPrev,
    read_only image2d_t targets, read_only image2d_t hiddenStatesPrev,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, int2 chunkSize, float weightAlpha)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
	
    int2 visiblePositionCenter = project(chunkPosition, hiddenToVisible);
	
	float target = read_imagef(targets, defaultSampler, hiddenPosition).x;
	float2 hiddenStatePrev = read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).xy;
	
	//float update = target * (1.0f - hiddenStatePrev.x) + (1.0f - target) * hiddenStatePrev.x;
	
    float error = target - sigmoid(hiddenStatePrev.y);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float visibleStatePrev = read_imagef(visibleStatesPrev, defaultSampler, visiblePosition).x;

				float weight = weightPrev + weightAlpha * error * visibleStatePrev;

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
            }
        }
}

void kernel plLearnPredWeightsQ(read_only image2d_t visibleStates,
    read_only image2d_t targets,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, int2 chunkSize, float weightAlpha, float tdError, float lambda)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
	
    int2 visiblePositionCenter = project(chunkPosition, hiddenToVisible);
	
    float target = read_imagef(targets, defaultSampler, hiddenPosition).x;

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float2 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

				float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

				float2 weight = (float2)(weightPrev.x + weightAlpha * tdError * weightPrev.y, fmax(lambda * weightPrev.y, target * visibleState));

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
            }
        }
}

void kernel plPropagate(read_only image2d_t hiddenStates, read_only image2d_t targetStates,
    read_only image2d_t visibleStatesBack, write_only image2d_t visibleStatesFront,
	read_only image3d_t weights,
    int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
    int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = project(visiblePosition, visibleToHidden);

	float sum = read_imagef(visibleStatesBack, defaultSampler, visiblePosition).x;

    float subSum = 0.0f;

    for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
        for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize)) {
                // Next layer node's receptive field
                int2 fieldCenter = project(hiddenPosition, hiddenToVisible);

                int2 fieldLowerBound = fieldCenter - (int2)(radius);
                int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds

                // Check for containment
                if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {
                    int2 offset = visiblePosition - fieldLowerBound;

                    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;
					float targetState = read_imagef(targetStates, defaultSampler, hiddenPosition).x;

                    int wi = offset.y + offset.x * (radius * 2 + 1);

                    float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                    subSum += (targetState - hiddenState) * weight;
                }
            }
        }

    write_imagef(visibleStatesFront, visiblePosition, (float4)(sum + subSum, 0.0f, 0.0f, 0.0f));
}