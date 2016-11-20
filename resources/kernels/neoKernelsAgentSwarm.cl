// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// -------------------------------------- Agent Layer ---------------------------------------

void kernel alActivate(read_only image2d_t hiddenStates,
    read_only image3d_t weights,
    read_only image2d_t hiddenSummationBack, write_only image2d_t hiddenSummationFront,
    int2 hiddenSize, float2 qToHidden, int radius)
{
    int2 qPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = project(qPosition, qToHidden);

    float sum = read_imagef(hiddenSummationBack, defaultSampler, qPosition).x;

    float q = 0.0f;

    int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize)) {
                int2 offset = hiddenPosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float weight = read_imagef(weights, defaultSampler, (int4)(qPosition.x, qPosition.y, wi, 0)).x;

                float state = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

                q += state * weight;
            }
        }

    write_imagef(hiddenSummationFront, qPosition, (float4)(sum + q, 0.0f, 0.0f, 0.0f));
}

void kernel alLearnQ(read_only image2d_t hiddenStates,
    read_only image2d_t qStates, read_only image2d_t qStatesPrev,
    read_only image2d_t tdErrors,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 hiddenSize, float2 qToHidden, int radius, float alpha, float lambda)
{
    int2 qPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = project(qPosition, qToHidden);

    float tdError = read_imagef(tdErrors, defaultSampler, qPosition).x;
    float qState = read_imagef(qStates, defaultSampler, qPosition).x;
    float qStatePrev = read_imagef(qStatesPrev, defaultSampler, qPosition).x;

    int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize)) {
                int2 offset = hiddenPosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float2 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(qPosition.x, qPosition.y, wi, 0)).xy;

                float state = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

                float2 weight = (float2)(weightPrev.x + alpha * fmin(1.0f, fmax(-1.0f, tdError)) * weightPrev.y, fmax(lambda * weightPrev.y, state));

                write_imagef(weightsFront, (int4)(qPosition.x, qPosition.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
            }
        }
}

void kernel alLearnActions(read_only image2d_t hiddenStates, read_only image2d_t actionProbabilities, 
    read_only image2d_t tdErrors, read_only image2d_t oneHotActions,
    read_only image3d_t weightsBack, write_only image3d_t weightsFront,
    int2 hiddenSize, float2 aToHidden, int radius, float alpha, float lambda, int2 subActionDims, float maxActionWeightMag)
{
    int2 aPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 hiddenPositionCenter = project(aPosition, aToHidden);

    float tdError = read_imagef(tdErrors, defaultSampler, (int2)(aPosition.x / subActionDims.x, aPosition.y / subActionDims.y)).x;
    float2 oneHotAction = read_imagef(oneHotActions, defaultSampler, aPosition).xy;

	float probability = read_imagef(actionProbabilities, defaultSampler, aPosition).x;
	
    int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

    for (int dx = -radius; dx <= radius; dx++)
        for (int dy = -radius; dy <= radius; dy++) {
            int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

            if (inBounds0(hiddenPosition, hiddenSize)) {
                int2 offset = hiddenPosition - fieldLowerBound;

                int wi = offset.y + offset.x * (radius * 2 + 1);

                float2 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(aPosition.x, aPosition.y, wi, 0)).xy;

                float state = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

                float2 weight = (float2)(weightPrev.x + alpha * tdError * weightPrev.y, lambda * weightPrev.y + oneHotAction.y * (1.0f - lambda) * (oneHotAction.x - probability) * state);

                write_imagef(weightsFront, (int4)(aPosition.x, aPosition.y, wi, 0), (float4)(fmin(maxActionWeightMag, fmax(-maxActionWeightMag, weight.x)), weight.y, 0.0f, 0.0f));
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

void kernel alGetAction(read_only image2d_t activations, write_only image2d_t probabilities,
	write_only image2d_t actions, int2 subActionDims, uint2 seed)
{
    uint2 seedValue = seed + (uint2)(get_global_id(0) * 73 + 2, get_global_id(1) * 45 + 12) * 44;
	
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float expSum = 0.0f;
	
	for (int x = 0; x < subActionDims.x; x++)
        for (int y = 0; y < subActionDims.y; y++) {
            float value = read_imagef(activations, defaultSampler, position * subActionDims + (int2)(x, y)).x;

            expSum += exp(value);
        }
	
    float select = randFloat(&seedValue);
	
	int selectIndex = 0;
	
	float sumSoFar = 0.0f;
	
	uchar selected = 0;
	
    for (int x = 0; x < subActionDims.x; x++)
        for (int y = 0; y < subActionDims.y; y++) {
			int2 subPosition = position * subActionDims + (int2)(x, y);
		
            float value = read_imagef(activations, defaultSampler, subPosition).x;

			float probability = exp(value) / expSum;
			
			write_imagef(probabilities, subPosition, (float4)(probability, 0.0f, 0.0f, 0.0f));
			
			sumSoFar += probability;
			
            if (!selected && sumSoFar >= select) {
				selectIndex = x + y * subActionDims.x;
					
				selected = 1;
			}
        }

    write_imagef(actions, position, (float4)(selectIndex));
}

void kernel alSetAction(read_only image2d_t modulator,
    read_only image2d_t actionsTaken, read_only image2d_t actionsTakenPrev,
    read_only image2d_t qStates, read_only image2d_t qStatesPrev, write_only image2d_t tdErrorsTrain, write_only image2d_t oneHotActions,
    int2 subActionDims, float reward, float gamma)
{
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float modulate = read_imagef(modulator, defaultSampler, position).x;

    float actionTaken = read_imagef(actionsTaken, defaultSampler, position).x;
    float actionTakenPrev = read_imagef(actionsTakenPrev, defaultSampler, position).x;

    int actionTakeni = (int)(round(actionTaken));
    int actionTakenPrevi = (int)(round(actionTakenPrev));

    float pred = read_imagef(qStates, defaultSampler, position).x;
    float predPrev = read_imagef(qStatesPrev, defaultSampler, position).x;

    float tdError = reward + gamma * pred - predPrev;
    //float tdError = reward - predPrev;

	write_imagef(tdErrorsTrain, position, (float4)(tdError, 0.0f, 0.0f, 0.0f));
            
    for (int x = 0; x < subActionDims.x; x++)
        for (int y = 0; y < subActionDims.y; y++) {
            int index = x + y * subActionDims.x;

            int2 subPosition = position * subActionDims + (int2)(x, y);
			
			float act = index == actionTakeni ? 1.0f : 0.0f;

            write_imagef(oneHotActions, subPosition, (float4)(act * modulate, modulate, 0.0f, 0.0f));
        }
}