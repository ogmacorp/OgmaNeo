// --------------------------------------------------------------------------
//	Ogma Toolkit(OTK)
//	Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
// --------------------------------------------------------------------------

// ----------------------------------------- Sparse Features -----------------------------------------

void kernel sfcAddSample(read_only image2d_t visibleStates,
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

void kernel sfcStimulus(read_only image3d_t samples,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, read_only image3d_t weights,
	int2 visibleSize, float2 hiddenToVisible, int2 chunkSize, float2 chunksToHidden, int radius, int numSamples, uchar ignoreMiddle)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
	float2 chunkCenterf = ((float2)(chunkPosition.x, chunkPosition.y) + (float2)(0.5f)) * chunksToHidden;
	int2 chunkCenter = (int2)(chunkCenterf.x, chunkCenterf.y);
	
	int2 visiblePositionCenter = project(chunkCenter, hiddenToVisible);
	
	float sum = read_imagef(hiddenSummationTempBack, defaultSampler, hiddenPosition).x;

    float subSum = 0.0f;
	
	float count = 0.0f;
	
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
		
					float delta = sample - weight;
		
					subSum += -delta * delta;
					count += 1.0f;
				}
			}
	}
		
    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum / fmax(0.0001f, count), 0.0f, 0.0f, 0.0f));
}

void kernel sfcActivate(read_only image2d_t hiddenStimuli, read_only image2d_t hiddenStatesPrev,
	write_only image2d_t hiddenActivationsFront)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float hiddenStimulus = read_imagef(hiddenStimuli, defaultSampler, hiddenPosition).x;
	
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(exp(hiddenStimulus), 0.0f, 0.0f, 0.0f));
}

void kernel sfcInhibit(read_only image2d_t activations,
	write_only image2d_t hiddenStatesFront,
	write_only image2d_t chunkWinners,
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
		
	write_imagei(chunkWinners, chunkPosition, (int4)(maxDelta.x, maxDelta.y, 0, 0));
		
    for (int dx = 0; dx < chunkSize.x; dx++)
		for (int dy = 0; dy < chunkSize.y; dy++) {
			int2 hiddenPosition = hiddenStartPosition + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				//float tracePrev = read_imagef(hiddenStatesBack, defaultSampler, hiddenPosition).y;
			
                if (dx == maxDelta.x && dy == maxDelta.y)
					write_imagef(hiddenStatesFront, hiddenPosition, (float4)(1.0f, 0.0f, 0.0f, 0.0f));
				else
					write_imagef(hiddenStatesFront, hiddenPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
			}
		}
}

void kernel sfcInhibitOther(read_only image2d_t activations,
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
				//float tracePrev = read_imagef(hiddenStatesBack, defaultSampler, hiddenPosition).y;
			
                if (dx == maxDelta.x && dy == maxDelta.y)
					write_imagef(hiddenStatesFront, hiddenPosition, (float4)(1.0f, 0.0f, 0.0f, 0.0f));
				else
					write_imagef(hiddenStatesFront, hiddenPosition, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
			}
		}
}

void kernel sfcReconstruct(read_only image2d_t hiddenStates, read_only image2d_t hiddenActivations,
    write_only image3d_t recons, read_only image3d_t weights,
    int2 visibleSize, int2 hiddenSize, float2 visibleToHidden, float2 hiddenToVisible, int2 chunkSize, float2 chunksToHidden, int radius, int2 reverseRadii, int numSamples)
{
    int2 visiblePosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = project(visiblePosition, visibleToHidden);

	// Find chunks for this input
    for (int s = 0; s < numSamples; s++) {
        float recon = 0.0f;

        for (int dx = -reverseRadii.x; dx <= reverseRadii.x; dx++)
            for (int dy = -reverseRadii.y; dy <= reverseRadii.y; dy++) {
                int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

                if (inBounds0(hiddenPosition, hiddenSize)) {
					int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
					float2 chunkCenterf = ((float2)(chunkPosition.x, chunkPosition.y) + (float2)(0.5f)) * chunksToHidden;
					int2 chunkCenter = (int2)(chunkCenterf.x, chunkCenterf.y);
				
                    // Next layer node's receptive field
                    int2 fieldCenter = project(chunkCenter, hiddenToVisible);

                    int2 fieldLowerBound = fieldCenter - (int2)(radius);
                    int2 fieldUpperBound = fieldCenter + (int2)(radius + 1); // So is included in inBounds

                    // Check for containment
                    if (inBounds(visiblePosition, fieldLowerBound, fieldUpperBound)) {
                        int2 offset = visiblePosition - fieldLowerBound;

                        float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;
						float hiddenActivation = read_imagef(hiddenActivations, defaultSampler, hiddenPosition).x;

                        int wi = s + numSamples * (offset.y + offset.x * (radius * 2 + 1));

                        float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                        recon += hiddenState * hiddenActivation * weight;
                    }
                }
            }

        write_imagef(recons, (int4)(visiblePosition.x, visiblePosition.y, s, 0), (float4)(recon, 0.0f, 0.0f, 0.0f));
    }
}

void kernel sfcLearnWeights(read_only image2d_t hiddenStates,
	read_only image2d_t chunkWinners,
    read_only image3d_t samples,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 hiddenSize, int2 visibleSize, float2 hiddenToVisible, int2 chunkSize, float2 chunksToHidden, int radius, float weightAlpha, int numSamples, float gamma)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);
	float2 chunkCenterf = ((float2)(chunkPosition.x, chunkPosition.y) + (float2)(0.5f)) * chunksToHidden;
	int2 chunkCenter = (int2)(chunkCenterf.x, chunkCenterf.y);
	
	int2 visiblePositionCenter = project(chunkCenter, hiddenToVisible);
	
	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

	int2 chunkWinner = read_imagei(chunkWinners, defaultSampler, chunkPosition).xy;
	
	int2 hiddenStartPosition = chunkPosition * chunkSize;
	
	int2 delta = (hiddenStartPosition + chunkWinner) - hiddenPosition;
	
	float strength = exp(-(delta.x * delta.x + delta.y * delta.y) * gamma);
	
	//float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;
	
	for (int s = 0; s < numSamples; s++) {
		for (int dx = -radius; dx <= radius; dx++)
			for (int dy = -radius; dy <= radius; dy++) {
				int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

				if (inBounds0(visiblePosition, visibleSize)) {
					int2 offset = visiblePosition - fieldLowerBound;

					int wi = s + numSamples * (offset.y + offset.x * (radius * 2 + 1));

					float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

					float sample = read_imagef(samples, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;
					//float recon = read_imagef(recons, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;
					
					float sLearn = strength * (sample - weightPrev); 

					write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weightPrev + weightAlpha * sLearn, 0.0f, 0.0f, 0.0f));
				}
			}
	}
}

void kernel sfcLearnBiases(read_only image2d_t hiddenStimuli, read_only image2d_t hiddenStates,
	read_only image2d_t biasesBack, write_only image2d_t biasesFront, float activeRatio, float biasAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;
	float hiddenStimulus = read_imagef(hiddenStimuli, defaultSampler, hiddenPosition).x;
	
	// Bias update
	float biasPrev = read_imagef(biasesBack, defaultSampler, hiddenPosition).x;
	
	write_imagef(biasesFront, hiddenPosition, (float4)(biasPrev + biasAlpha * (activeRatio - hiddenState)));
}

void kernel sfcDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront, float lambda) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float input = read_imagef(inputs, defaultSampler, position).x;
	
	float tracePrev = read_imagef(outputsBack, defaultSampler, position).y;
	
	write_imagef(outputsFront, position, (float4)(input, lambda * tracePrev + (1.0f - lambda) * input, 0.0f, 0.0f));
}