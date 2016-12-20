// --------------------------------------------------------------------------
//	Ogma Toolkit(OTK)
//	Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
// --------------------------------------------------------------------------

// ----------------------------------------- Sparse Features -----------------------------------------

void kernel sfrAddSample(read_only image2d_t visibleStates,
    read_only image3d_t samplesBack, write_only image3d_t samplesFront,
    int numSamples)
{
    int2 position = (int2)(get_global_id(0), get_global_id(1));
    
    float visibleState = read_imagef(visibleStates, defaultSampler, position).x;

    for (int s = 1; s < numSamples; s++) {
        float samplePrev = read_imagef(samplesBack, defaultSampler, (int4)(position.x, position.y, s - 1, 0)).x;

        write_imagef(samplesFront, (int4)(position.x, position.y, s, 0), (float4)(samplePrev, 0.0f, 0.0f, 0.0f));
    }

    write_imagef(samplesFront, (int4)(position.x, position.y, 0, 0), (float4)(visibleState, 0.0f, 0.0f, 0.0f));
}

void kernel sfrStimulus(read_only image3d_t samples,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront,
	read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int radius, int numSamples, uchar ignoreMiddle)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);
	
	float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;

    float subSum = 0.0f;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int s = 0; s < numSamples; s++) {
		for (int dx = -radius; dx <= radius; dx++)
			for (int dy = -radius; dy <= radius; dy++) {
				int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

				if (ignoreMiddle && dx == 0 && dy == 0)
					continue;
				
				if (inBounds0(visiblePosition, visibleSize)) {
					int2 offset = visiblePosition - fieldLowerBound;
	
					int wi = s + numSamples * (offset.y + offset.x * (radius * 2 + 1));

					float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

					float sample = read_imagef(samples, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;

					subSum += sample * weight;
				}
			}
	}
		
    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum, 0.0f, 0.0f, 0.0f));
}

void kernel sfrInhibit(read_only image2d_t activations,
    read_only image2d_t hiddenStatesBack, write_only image2d_t hiddenStatesFront,
    int2 hiddenSize, int radius, float activeRatio, float gamma)
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
	
	float2 hiddenStatePrev = read_imagef(hiddenStatesBack, defaultSampler, hiddenPosition).xy;

    write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state * fmax(0.0f, tanh(activation)), fmax(gamma * hiddenStatePrev.y, hiddenStatePrev.x), 0.0f, 0.0f));
}

void kernel sfrInhibitOther(read_only image2d_t activations,
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

    write_imagef(hiddenStatesFront, hiddenPosition, (float4)(state * fmax(0.0f, tanh(activation)), 0.0f, 0.0f, 0.0f));
}

void kernel sfrPredict(read_only image2d_t visibleStates,
	read_only image3d_t weights, write_only image2d_t predictions,
	int2 visibleSize, float2 hiddenToVisible, int radius)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);
	
	float sum = 0.0f;

	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);
			
			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

				sum += visibleState * weight;
			}
		}
		
    write_imagef(predictions, hiddenPosition, (float4)(sum, 0.0f, 0.0f, 0.0f));
}

void kernel sfrErrorProp(read_only image2d_t visibleStates, read_only image2d_t predictionsPrev,
	read_only image2d_t hiddenStates,
    read_only image2d_t hiddenErrorSummationTempBack, write_only image2d_t hiddenErrorSummationTempFront, 
	read_only image3d_t weights,
    int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int radius, int2 reverseRadii)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    float error = read_imagef(hiddenErrorSummationTempBack, defaultSampler, hiddenPosition).x;

	float subError = 0.0f;

    for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
        for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                // Next layer node's receptive field
                int2 fieldCenter = project(visiblePosition, visibleToHidden);

                int2 fieldLowerBound = fieldCenter - (int2)(radius);
                int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds

                // Check for containment
                if (inBounds(hiddenPosition, fieldLowerBound, fieldUpperBound)) {
                    int2 offset = hiddenPosition - fieldLowerBound;

                    float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;
					float predictionPrev = read_imagef(predictionsPrev, defaultSampler, visiblePosition).x;

                    int wi = offset.y + offset.x * (radius * 2 + 1);

                    float weight = read_imagef(weights, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, wi, 0)).x;

                    subError += weight * (visibleState - predictionPrev);
                }
            }
        }

    write_imagef(hiddenErrorSummationTempFront, hiddenPosition, (float4)(error + subError, 0.0f, 0.0f, 0.0f));
}

void kernel sfrLearnWeightsHidden(read_only image2d_t hiddenStatesPrev,
	read_only image2d_t errors,
    read_only image3d_t samplesPrev,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha, int numSamples, float activeRatio)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);
	
	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	float2 hiddenStatePrev = read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).xy;
	
	float error = (hiddenStatePrev.x == 0.0f ? 0.0f : 1.0f - hiddenStatePrev.x * hiddenStatePrev.x) * (read_imagef(errors, defaultSampler, hiddenPosition).x > 0.0f ? 1.0f : -1.0f);
	
	error *= (error > 0.0f ? (1.0f - hiddenStatePrev.y) : hiddenStatePrev.y);
	
	for (int s = 0; s < numSamples; s++) {
		for (int dx = -radius; dx <= radius; dx++)
			for (int dy = -radius; dy <= radius; dy++) {
				int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

				if (inBounds0(visiblePosition, visibleSize)) {
					int2 offset = visiblePosition - fieldLowerBound;

					int wi = s + numSamples * (offset.y + offset.x * (radius * 2 + 1));

					float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

					float samplePrev = read_imagef(samplesPrev, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;

					float sLearn = error * samplePrev; 

					write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weightPrev + weightAlpha * sLearn, 0.0f, 0.0f, 0.0f));
				}
			}
	}
}

void kernel sfrLearnWeightsVisible(read_only image2d_t visibleStates,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	read_only image2d_t hiddenStates, read_only image2d_t predictionsPrev,
	int2 visibleSize, float2 hiddenToVisible, int radius, float weightAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);
	
	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	float error = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x - read_imagef(predictionsPrev, defaultSampler, hiddenPosition).x;
	
	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);
			
			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

				float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

				float weight = weightPrev + weightAlpha * error * visibleState;
				
				write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
			}
		}
}

void kernel sfrLearnBiases(read_only image2d_t hiddenStatesPrev,
	read_only image2d_t errors,
	read_only image2d_t biasesBack, write_only image2d_t biasesFront,
	float activeRatio, float gamma, float biasAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float2 hiddenStatePrev = read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).xy;
	
	float error = (hiddenStatePrev.x == 0.0f ? 0.0f : 1.0f - hiddenStatePrev.x * hiddenStatePrev.x) * (read_imagef(errors, defaultSampler, hiddenPosition).x > 0.0f ? 1.0f : -1.0f);
	
	error *= (error > 0.0f ? (1.0f - hiddenStatePrev.y) : hiddenStatePrev.y);
	
	float biasPrev = read_imagef(biasesBack, defaultSampler, hiddenPosition).x;
	
	float bias = biasPrev + biasAlpha * error;
	
	write_imagef(biasesFront, hiddenPosition, (float4)(bias, 0.0f, 0.0f, 0.0f));
}

void kernel sfrDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront, float lambda) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float input = read_imagef(inputs, defaultSampler, position).x;
		
	float tracePrev = read_imagef(outputsBack, defaultSampler, position).y;
		
	float trace = lambda * tracePrev + (1.0f - lambda) * input;
		
	write_imagef(outputsFront, position, (float4)(input - tracePrev, trace, 0.0f, 0.0f));
}