// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// -------------------------------------- Agent Layer ---------------------------------------

void kernel alDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float input = read_imagef(inputs, defaultSampler, position).x;

	float outputPrev = read_imagef(outputsBack, defaultSampler, position).x;

    write_imagef(outputsFront, position, (float4)(input, fabs(input - outputPrev), 0.0f, 0.0f));
}

void kernel alActivate(read_only image2d_t visibleStates,
    read_only image3d_t weights,
    read_only image2d_t hiddenSummationBack, write_only image2d_t hiddenSummationFront,
    int2 visibleSize, float2 hiddenToVisible, int radius)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    float sum = read_imagef(hiddenSummationBack, defaultSampler, hiddenPosition).x;

    float q = 0.0f;
	float stateSum = 0.0f;

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weight = read_imagef(weights, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float state = read_imagef(visibleStates, defaultSampler, visiblePosition).x;

                q += state * weight;
				stateSum += state;
            }
        }

    write_imagef(hiddenSummationFront, hiddenPosition, (float4)(sum + q / fmax(0.0001f, stateSum), 0.0f, 0.0f, 0.0f));
}

void kernel alLearnQ(read_only image2d_t visibleStates,
    read_only image2d_t oneHotActions, read_only image2d_t tdErrors,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 visibleSize, float2 hiddenToVisible, int radius, float alpha, float lambda, int2 subActionDims)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	float oneHotAction = read_imagef(oneHotActions, defaultSampler, hiddenPosition).x;
	
	float tdError = read_imagef(tdErrors, defaultSampler, (int2)(hiddenPosition.x / subActionDims.x, hiddenPosition.y / subActionDims.y)).x;
			
	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

    int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);		
	
    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

            if (inBounds0(visiblePosition, visibleSize)) {
                int2 offset = visiblePosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float2 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).xy;

                float2 state = read_imagef(visibleStates, defaultSampler, visiblePosition).xy;

                float2 weight = (float2)(weightPrev.x + alpha * tdError * weightPrev.y, fmax(lambda * weightPrev.y, oneHotAction * state.x * state.y));

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
            }
        }
}

void kernel alActionToOneHot(read_only image2d_t hiddenStates, read_only image2d_t actions, write_only image2d_t oneHotActions, int2 subActionDims, uchar modulate) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float hiddenState = modulate ? read_imagef(hiddenStates, defaultSampler, position).x : 1.0f;

    float action = read_imagef(actions, defaultSampler, position).x;

    int actioni = (int)(round(action));

    int2 actionPosition = position * subActionDims + (int2)(actioni % subActionDims.x, actioni / subActionDims.x);

    for (int x = 0; x < subActionDims.x; x++)
        for (int y = 0; y < subActionDims.y; y++) {
            int index = x + y * subActionDims.x;

            int2 subPosition = position * subActionDims + (int2)(x, y);

            write_imagef(oneHotActions, subPosition, (float4)(index == actioni ? hiddenState : 0.0f));
        }
}

void kernel alGetAction(read_only image2d_t activations,
	write_only image2d_t actionsTaken, write_only image2d_t actionsTakenMax, int2 subActionDims, float epsilon, uint2 seed)
{
    uint2 seedValue = seed + (uint2)(get_global_id(0) * 73 + 2, get_global_id(1) * 45 + 12) * 44;
	
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int maxIndex = 0;
	float maxValue = -99999.0f;
	
	for (int x = 0; x < subActionDims.x; x++)
        for (int y = 0; y < subActionDims.y; y++) {
            float value = read_imagef(activations, defaultSampler, position * subActionDims + (int2)(x, y)).x;

            if (value > maxValue) {
				maxValue = value;
				
				maxIndex = x + y * subActionDims.x;
			}
        }
		
	int exploreIndex = maxIndex;
	
    if (randFloat(&seedValue) < epsilon)
		exploreIndex = (int)(randFloat(&seedValue) * (subActionDims.x * subActionDims.y));

    write_imagef(actionsTaken, position, (float4)(exploreIndex, 0.0f, 0.0f, 0.0f));
	write_imagef(actionsTakenMax, position, (float4)(maxIndex, 0.0f, 0.0f, 0.0f));
}

void kernel alSetAction(read_only image2d_t modulator,
    read_only image2d_t actionsTaken, read_only image2d_t actionsTakenPrev,
    read_only image2d_t actionsTakenMax, read_only image2d_t actionsTakenMaxPrev,
    read_only image2d_t qStates, read_only image2d_t qStatesPrev, write_only image2d_t tdErrorsTrain, write_only image2d_t oneHotActions,
    int2 subActionDims, int2 chunkSize, float reward, float gamma)
{
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float modulate = read_imagef(modulator, defaultSampler, position).x;

    float actionTaken = read_imagef(actionsTaken, defaultSampler, position).x;
    float actionTakenPrev = read_imagef(actionsTakenPrev, defaultSampler, position).x;
	float actionTakenMax = read_imagef(actionsTakenMax, defaultSampler, position).x;
    float actionTakenMaxPrev = read_imagef(actionsTakenMaxPrev, defaultSampler, position).x;

    int actionTakeni = (int)(round(actionTaken));
    int actionTakenPrevi = (int)(round(actionTakenPrev));
	int actionTakenMaxi = (int)(round(actionTakenMax));
    int actionTakenMaxPrevi = (int)(round(actionTakenMaxPrev));
	
	int2 actionTakenPosition = position * subActionDims + (int2)(actionTakeni % subActionDims.x, actionTakeni / subActionDims.x);
	int2 actionTakenPrevPosition = position * subActionDims + (int2)(actionTakenPrevi % subActionDims.x, actionTakenPrevi / subActionDims.x);
	int2 actionTakenMaxPosition = position * subActionDims + (int2)(actionTakenMaxi % subActionDims.x, actionTakenMaxi / subActionDims.x);
	int2 actionTakenMaxPrevPosition = position * subActionDims + (int2)(actionTakenMaxPrevi % subActionDims.x, actionTakenMaxPrevi / subActionDims.x);
	
    float pred = read_imagef(qStates, defaultSampler, actionTakenPosition).x;
    float predPrev = read_imagef(qStatesPrev, defaultSampler, actionTakenPrevPosition).x;

    float tdError = reward + gamma * pred - predPrev;
    //float tdError = reward - predPrev;

	write_imagef(tdErrorsTrain, position, (float4)(tdError, 0.0f, 0.0f, 0.0f));
            
    for (int x = 0; x < subActionDims.x; x++)
        for (int y = 0; y < subActionDims.y; y++) {
            int index = x + y * subActionDims.x;

            int2 subPosition = position * subActionDims + (int2)(x, y);
			
			float act = (index == actionTakeni) ? 1.0f : 0.0f;

            write_imagef(oneHotActions, subPosition, (float4)(act * modulate, 0.0f, 0.0f, 0.0f));
        }
}

void kernel alSpread(read_only image2d_t oneHotActions, write_only image2d_t spreadStates,
    int2 numActionTiles, int2 subActionDims,
	float chunkGamma, int2 chunkSize)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
	/*int2 actionTilePosition = (int2)(hiddenPosition.x / subActionDims.x, hiddenPosition.y / subActionDims.y);
	
	int2 chunkPosition = (int2)(actionTilePosition.x / chunkSize.x, actionTilePosition.y / chunkSize.y);

	float strength = 0.0f;

	int2 thisActionDelta = hiddenPosition - actionTilePosition * subActionDims;

	// Search for influences
	int2 chunkStartPosition = chunkPosition * chunkSize;
	
	for (int dx = 0; dx < chunkSize.x; dx++)
		for (int dy = 0; dy < chunkSize.y; dy++) {
			int2 otherActionTilePosition = chunkStartPosition + (int2)(dx, dy);

			if (inBounds0(otherActionTilePosition, numActionTiles)) {	
				int2 delta = otherActionTilePosition - actionTilePosition;
			
				float falloff = exp(-(delta.x * delta.x + delta.y * delta.y) * chunkGamma);
			
				// Search one hot actions
				int2 oneHotPosition = otherActionTilePosition * subActionDims + thisActionDelta;
				
				float oneHotAction = read_imagef(oneHotActions, defaultSampler, oneHotPosition).x;
			
				strength += falloff * oneHotAction;
			}
		}*/
		
	float oneHotAction = read_imagef(oneHotActions, defaultSampler, hiddenPosition).x;
		
	write_imagef(spreadStates, hiddenPosition, (float4)(oneHotAction, 0.0f, 0.0f, 0.0f));
}