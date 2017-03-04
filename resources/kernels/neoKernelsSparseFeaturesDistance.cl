// --------------------------------------------------------------------------
//	Ogma Toolkit(OTK)
//	Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
// --------------------------------------------------------------------------

// ----------------------------------------- Sparse Features -----------------------------------------

void kernel sfdAddSample(read_only image2d_t visibleStates,
    read_only image3d_t samplesBack, write_only image3d_t samplesFront,
    int numSamples)
{
    int2 position = (int2)(get_global_id(0), get_global_id(1));
    
    float visibleState = read_imagef(visibleStates, defaultSampler, position).x;

    for (int s = numSamples - 1; s > 0; s--) {
        float samplePrev = read_imagef(samplesBack, defaultSampler, (int4)(position.x, position.y, s - 1, 0)).x;

        write_imagef(samplesFront, (int4)(position.x, position.y, s, 0), (float4)(samplePrev, 0.0f, 0.0f, 0.0f));
    }

    write_imagef(samplesFront, (int4)(position.x, position.y, 0, 0), (float4)(visibleState, 0.0f, 0.0f, 0.0f));
}

void kernel sfdStimulus(read_only image3d_t samples,
	read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront,
	read_only image3d_t weights,
	int2 visibleSize, float2 chunkToVisible, int2 chunkSize, int radius, int numSamples, uchar ignoreMiddle)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);

	int2 visiblePositionCenter = project(chunkPosition, chunkToVisible);

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
					
					float delta = sample - weight;
					
					subSum += -delta * delta;
				}
			}
	}
		
    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum, 0.0f, 0.0f, 0.0f));
}

void kernel sfdActivate(read_only image2d_t hiddenStimuli, read_only image2d_t hiddenStatesPrev,
	write_only image2d_t hiddenActivationsFront)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float hiddenStimulus = read_imagef(hiddenStimuli, defaultSampler, hiddenPosition).x;
	
	float tracePrev = read_imagef(hiddenStatesPrev, defaultSampler, hiddenPosition).y;
	
	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(hiddenStimulus, 0.0f, 0.0f, 0.0f)); // + 1.0f / fmax(1.0f, tracePrev)
}

void kernel sfdInhibit(read_only image2d_t activations,
	read_only image2d_t hiddenStatesBack,
	write_only image2d_t hiddenStatesFront,
	write_only image2d_t chunkWinners,
	int2 hiddenSize, int2 chunkSize, float gamma)
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
		
	write_imagef(chunkWinners, chunkPosition, (float4)((float)maxDelta.x, (float)maxDelta.y, 0.0f, 0.0f));

    for (int dx = 0; dx < chunkSize.x; dx++)
		for (int dy = 0; dy < chunkSize.y; dy++) {
			int2 hiddenPosition = hiddenStartPosition + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				float tracePrev = read_imagef(hiddenStatesBack, defaultSampler, hiddenPosition).y;

				float neighbor = (abs(maxDelta.x - dx) + abs(maxDelta.y - dy)) <= 1 ? 1.0f : 0.0f;
				
				float hiddenState = (dx == maxDelta.x && dy == maxDelta.y) ? 1.0f : 0.0f;

				write_imagef(hiddenStatesFront, hiddenPosition, (float4)(hiddenState, fmin(99999.0f, tracePrev * gamma + neighbor), 0.0f, 0.0f));
			}
		}
}

void kernel sfdInhibitOther(read_only image2d_t activations,
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
                float hiddenState = (dx == maxDelta.x && dy == maxDelta.y) ? 1.0f : 0.0f;
                
				write_imagef(hiddenStatesFront, hiddenPosition, (float4)(hiddenState, 0.0f, 0.0f, 0.0f));
			}
		}
}

void kernel sfdLearnWeights(read_only image2d_t chunkWinners,
	read_only image2d_t hiddenStates,
    read_only image3d_t samples,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 hiddenSize, int2 visibleSize, float2 chunkToVisible, int2 chunkSize, int radius, float weightAlpha, int numSamples)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	int2 chunkPosition = (int2)(hiddenPosition.x / chunkSize.x, hiddenPosition.y / chunkSize.y);

	int2 visiblePositionCenter = project(chunkPosition, chunkToVisible);
	
	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
	float2 chunkWinnerf = read_imagef(chunkWinners, defaultSampler, chunkPosition).xy;
	
	int2 chunkWinner = (int2)(chunkWinnerf.x, chunkWinnerf.y);
	
	int2 delta = chunkPosition * chunkSize + chunkWinner - hiddenPosition;
	
	float trace = read_imagef(hiddenStates, defaultSampler, hiddenPosition).y;
	
	float update = (abs(delta.x) + abs(delta.y)) <= 1 ? 1.0f / fmax(1.0f, trace) : 0.0f;
	
	for (int s = 0; s < numSamples; s++) {
		for (int dx = -radius; dx <= radius; dx++)
			for (int dy = -radius; dy <= radius; dy++) {
				int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

				if (inBounds0(visiblePosition, visibleSize)) {
					int2 offset = visiblePosition - fieldLowerBound;
					
					int wi = s + numSamples * (offset.y + offset.x * (radius * 2 + 1));

					float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

					float sample = read_imagef(samples, defaultSampler, (int4)(visiblePosition.x, visiblePosition.y, s, 0)).x;
					
					float weight = weightPrev + weightAlpha * update * (sample - weightPrev);
					
					write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight, 0.0f, 0.0f, 0.0f));
				}
			}
	}
}

void kernel sfdDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront, float lambda) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float input = read_imagef(inputs, defaultSampler, position).x;
		
	float tracePrev = read_imagef(outputsBack, defaultSampler, position).y;
		
	float trace = lambda * tracePrev + (1.0f - lambda) * input;
		
	write_imagef(outputsFront, position, (float4)(input, trace, 0.0f, 0.0f));
}

void kernel sfdSum(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float input = read_imagef(inputs, defaultSampler, position).x;
		
	float sumPrev = read_imagef(outputsBack, defaultSampler, position).x;
		
	write_imagef(outputsFront, position, (float4)(sumPrev + input, 0.0f, 0.0f, 0.0f));
}

void kernel sfdSlice(read_only image3d_t samples,
    write_only image2d_t slice,
    int index)
{
    int2 position = (int2)(get_global_id(0), get_global_id(1));
    
    float sample = read_imagef(samples, defaultSampler, (int4)(position.x, position.y, index, 0)).x;

    write_imagef(slice, position, (float4)(sample, 0.0f, 0.0f, 0.0f));
}