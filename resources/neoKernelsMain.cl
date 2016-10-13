// --------------------------------------------------------------------------
//	Ogma Toolkit(OTK)
//	Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
// --------------------------------------------------------------------------

// ----------------------------------------- Samplers -----------------------------------------

constant sampler_t defaultSampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t normalizedClampedToEdgeNearestSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t unnormalizedClampedNearestSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP |
	CLK_FILTER_NEAREST;

constant sampler_t defaultNormalizedSampler = CLK_NORMALIZED_COORDS_TRUE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

constant sampler_t defaultUnnormalizedSampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_CLAMP_TO_EDGE |
	CLK_FILTER_NEAREST;

// ----------------------------------------- Common -----------------------------------------

float randFloat(uint2* state) {
	const float invMaxInt = 1.0f / 4294967296.0f;
	uint x = (*state).x * 17 + (*state).y * 13123;
	(*state).x = (x << 13) ^ x;
	(*state).y ^= (x << 7);

	uint tmp = x * (x * x * 15731 + 74323) + 871483;

	return convert_float(tmp) * invMaxInt;
}

float randNormal(uint2* state) {
	float u1 = randFloat(state);
	float u2 = randFloat(state);

	return sqrt(-2.0f * log(u1)) * cos(6.28318f * u2);
}

float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

float relu(float x, float leak) {
	x += 0.5f;

	if (x > 1.0f)
		return 1.0f + (x - 1.0f) * leak;

	return x > 0.0f ? x : x * leak;
}

float relud(float x, float leak) {
	x += 0.5f;

	return x > 0.0f && x < 1.0f ? 1.0f : leak;
}

bool inBounds0(int2 position, int2 upperBound) {
	return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
}

bool inBounds(int2 position, int2 lowerBound, int2 upperBound) {
	return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
}

int2 project(int2 position, float2 toScalars) {
    return (int2)(position.x * toScalars.x + 0.5f, position.y * toScalars.y + 0.5f);
}

// Initialize a random uniform 2D image (X field)
void kernel randomUniform2D(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;

	write_imagef(values, position, (float4)(value, 0.0f, 0.0f, 0.0f));
}

// Initialize a random uniform 3D image (X field)
void kernel randomUniform3D(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float value = randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x;

	write_imagef(values, (int4)(position, 0), (float4)(value, 0.0f, 0.0f, 0.0f));
}

// Initialize a random uniform 2D image (XY fields)
void kernel randomUniform2DXY(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 15 + 66, get_global_id(1) * 61 + 2) * 56;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, position, (float4)(v.x, v.y, 0.0f, 0.0f));
}

// Initialize a random uniform 2D image (XYZ fields)
void kernel randomUniform2DXYZ(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 15 + 66, get_global_id(1) * 61 + 2) * 56;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float3 v = (float3)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, position, (float4)(v.x, v.y, v.z, 0.0f));
}

// Initialize a random uniform 2D image (XZ fields)
void kernel randomUniform2DXZ(write_only image2d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, position, (float4)(v.x, 0.0f, v.y, 0.0f));
}

// Initialize a random uniform 3D image (XY fields)
void kernel randomUniform3DXY(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, (int4)(position, 0), (float4)(v.x, v.y, 0.0f, 0.0f));
}

// Initialize a random uniform 3D image (XZ fields)
void kernel randomUniform3DXZ(write_only image3d_t values, uint2 seed, float2 minMax) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

	int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

	float2 v = (float2)(randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x, randFloat(&seedValue) * (minMax.y - minMax.x) + minMax.x);

	write_imagef(values, (int4)(position, 0), (float4)(v.x, 0.0f, v.y, 0.0f));
}

// ----------------------------------------- Sparse Predictor -----------------------------------------

void kernel spStimulus(read_only image2d_t visibleStates,
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
				stateSum += visibleState;
			}
		}

    write_imagef(hiddenSummationTempFront, hiddenPosition, (float4)(sum + subSum / fmax(0.0001f, stateSum), 0.0f, 0.0f, 0.0f));
}

void kernel spActivate(read_only image2d_t stimuli, read_only image2d_t hiddenStates,  read_only image2d_t biases,
    read_only image2d_t hiddenActivationsBack, write_only image2d_t hiddenActivationsFront)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	
    float stimulus = read_imagef(stimuli, defaultSampler, hiddenPosition).x;

    float activationPrev = read_imagef(hiddenActivationsBack, defaultSampler, hiddenPosition).x;

    float statePrev = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

	float bias = read_imagef(biases, defaultSampler, hiddenPosition).x;
	
    float activation = stimulus + bias;

	write_imagef(hiddenActivationsFront, hiddenPosition, (float4)(activation, 0.0f, 0.0f, 0.0f));
}

void kernel spInhibit(read_only image2d_t activations,
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

void kernel spLearnWeights(read_only image2d_t hiddenStates,
    read_only image2d_t visibleStates,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront,
	int2 visibleSize, float2 hiddenToVisible, int radius, float activeRatio, float weightAlpha)
{
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);
	
	int2 fieldLowerBound = visiblePositionCenter - (int2)(radius);
	
    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

    for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 visiblePosition = visiblePositionCenter + (int2)(dx, dy);

			if (inBounds0(visiblePosition, visibleSize)) {
				int2 offset = visiblePosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0)).x;

                float visibleState = read_imagef(visibleStates, defaultSampler, visiblePosition).x;
				
                float learn = hiddenState * (visibleState - weightPrev);

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(fmin(1.0f, fmax(0.0f, weightPrev + weightAlpha * learn)), 0.0f, 0.0f, 0.0f));
            }
		}
}

void kernel spLearnBiases(read_only image2d_t stimuli, read_only image2d_t hiddenStates, read_only image2d_t hiddenBiasesBack, write_only image2d_t hiddenBiasesFront, float activeRatio, float biasAlpha) {
	int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));

	float stimulus = read_imagef(stimuli, defaultSampler, hiddenPosition).x;
    float hiddenState = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

	float hiddenBiasPrev = read_imagef(hiddenBiasesBack, defaultSampler, hiddenPosition).x;

	write_imagef(hiddenBiasesFront, hiddenPosition, (float4)(hiddenBiasPrev + biasAlpha * (-stimulus - hiddenBiasPrev), 0.0f, 0.0f, 0.0f));
}

void kernel spDeriveInputs(read_only image2d_t inputs, read_only image2d_t outputsBack, write_only image2d_t outputsFront) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));

	float input = read_imagef(inputs, defaultSampler, position).x;

	write_imagef(outputsFront, position, (float4)(input, 0.0f, 0.0f, 0.0f));
}

// ------------------------------------------- Predictor -------------------------------------------

void kernel plStimulus(read_only image2d_t visibleStates,
    read_only image2d_t hiddenSummationTempBack, write_only image2d_t hiddenSummationTempFront, 
    read_only image3d_t weights,
    int2 visibleSize, float2 hiddenToVisible, int radius)
{
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    int2 visiblePositionCenter = project(hiddenPosition, hiddenToVisible);

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

				float visibleStatePrev = read_imagef(visibleStatesPrev, defaultSampler, visiblePosition).x;

				float weight = weightPrev + alpha * error * visibleStatePrev;

                write_imagef(weightsFront, (int4)(hiddenPosition.x, hiddenPosition.y, wi, 0), (float4)(weight));
			}
		}
}

void kernel plThreshold(read_only image2d_t stimuli, write_only image2d_t thresholded) {
    int2 hiddenPosition = (int2)(get_global_id(0), get_global_id(1));
    
    float stimulus = read_imagef(stimuli, defaultSampler, hiddenPosition).x;

    write_imagef(thresholded, hiddenPosition, (float4)(stimulus > 0.5f ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));
}

// --------------------------------------------- Predictor ----------------------------------------------

void kernel pError(read_only image2d_t targets, read_only image2d_t predictionsPrev,
    write_only image2d_t errors)
{
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float target = read_imagef(targets, defaultSampler, position).x;

    float predictionPrev = read_imagef(predictionsPrev, defaultSampler, position).x;

    write_imagef(errors, position, (float4)(target - predictionPrev));
}

// -------------------------------------- Agent Layer ---------------------------------------

void kernel alFindQ(read_only image2d_t hiddenStates,
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

				float2 weight = read_imagef(weights, defaultSampler, (int4)(qPosition.x, qPosition.y, wi, 0)).xy;

				float state = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

				q += state * weight.x;
			}
		}

	write_imagef(hiddenSummationFront, qPosition, (float4)(sum + q, 0.0f, 0.0f, 0.0f));
}

void kernel alLearnQ(read_only image2d_t hiddenStates,
    read_only image2d_t qStates,  read_only image2d_t qStatesPrev,
    read_only image2d_t tdErrors, read_only image2d_t oneHotActions,
	read_only image3d_t weightsBack, write_only image3d_t weightsFront, 
	int2 hiddenSize, float2 qToHidden, int radius, float alpha, float lambda)
{
	int2 qPosition = (int2)(get_global_id(0), get_global_id(1));
	int2 hiddenPositionCenter = project(qPosition, qToHidden);
	
	float tdError = read_imagef(tdErrors, defaultSampler, qPosition).x;
    float oneHotAction = read_imagef(oneHotActions, defaultSampler, qPosition).x;
    float2 qState = read_imagef(qStates, defaultSampler, qPosition).xy;
    float2 qStatePrev = read_imagef(qStatesPrev, defaultSampler, qPosition).xy;

	int2 fieldLowerBound = hiddenPositionCenter - (int2)(radius);

	for (int dx = -radius; dx <= radius; dx++)
		for (int dy = -radius; dy <= radius; dy++) {
			int2 hiddenPosition = hiddenPositionCenter + (int2)(dx, dy);

			if (inBounds0(hiddenPosition, hiddenSize)) {
				int2 offset = hiddenPosition - fieldLowerBound;

				int wi = offset.y + offset.x * (radius * 2 + 1);

				float2 weightPrev = read_imagef(weightsBack, defaultSampler, (int4)(qPosition.x, qPosition.y, wi, 0)).xy;

				float state = read_imagef(hiddenStates, defaultSampler, hiddenPosition).x;

				float2 weight = (float2)(weightPrev.x + alpha * tdError * weightPrev.y, lambda * weightPrev.y + (1.0f - lambda) * oneHotAction * state);

                write_imagef(weightsFront, (int4)(qPosition.x, qPosition.y, wi, 0), (float4)(weight.x, weight.y, 0.0f, 0.0f));
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

void kernel alGetAction(read_only image2d_t predictions, write_only image2d_t actions, int2 subActionDims) {
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
	int maxIndex = 0;
	float maxValue = -99999.0f;

	for (int x = 0; x < subActionDims.x; x++)
		for (int y = 0; y < subActionDims.y; y++) {
			float value = read_imagef(predictions, defaultSampler, position * subActionDims + (int2)(x, y)).x;

			if (value > maxValue) {
				maxValue = value;
				maxIndex = x + y * subActionDims.x;
			}
		}
	
	write_imagef(actions, position, (float4)(maxIndex));
}

void kernel alSetAction(read_only image2d_t modulator,
    read_only image2d_t actions, read_only image2d_t actionsPrev,
    read_only image2d_t actionsTaken, read_only image2d_t actionsTakenPrev,
    read_only image2d_t predictions, read_only image2d_t predictionsPrev, write_only image2d_t tdErrorsTrain, write_only image2d_t oneHotActions,
    int2 subActionDims, float reward, float gamma)
{
	int2 position = (int2)(get_global_id(0), get_global_id(1));
	
    float modulate = read_imagef(modulator, defaultSampler, position).x;

	float action = read_imagef(actions, defaultSampler, position).x;
    float actionPrev = read_imagef(actionsPrev, defaultSampler, position).x;
    float actionTaken = read_imagef(actionsTaken, defaultSampler, position).x;
	float actionTakenPrev = read_imagef(actionsTakenPrev, defaultSampler, position).x;

	int actioni = (int)(round(action));
    int actionPrevi = (int)(round(actionPrev));
    int actionTakeni = (int)(round(actionTaken));
	int actionTakenPrevi = (int)(round(actionTakenPrev));

	int2 actionPosition = position * subActionDims + (int2)(actioni % subActionDims.x, actioni / subActionDims.x);
    int2 actionPrevPosition = position * subActionDims + (int2)(actionPrevi % subActionDims.x, actionPrevi / subActionDims.x);
    int2 actionTakenPosition = position * subActionDims + (int2)(actionTakeni % subActionDims.x, actionTakeni / subActionDims.x);
	int2 actionTakenPrevPosition = position * subActionDims + (int2)(actionTakenPrevi % subActionDims.x, actionTakenPrevi / subActionDims.x);

    float pred = read_imagef(predictions, defaultSampler, actionTakenPosition).x;
	float predPrev = read_imagef(predictionsPrev, defaultSampler, actionTakenPrevPosition).x;

    float tdError = reward + gamma * pred - predPrev;

	for (int x = 0; x < subActionDims.x; x++)
		for (int y = 0; y < subActionDims.y; y++) {
			int index = x + y * subActionDims.x;

			int2 subPosition = position * subActionDims + (int2)(x, y);

			write_imagef(tdErrorsTrain, subPosition, (float4)(tdError));
			write_imagef(oneHotActions, subPosition, (float4)(index == actionTakeni ? modulate : 0.0f));
		}
}

void kernel alActionExploration(read_only image2d_t actions, write_only image2d_t actionsExploratory, float epsilon, int subActionCount, uint2 seed) {
	uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 43, get_global_id(1) * 21 + 42) * 12;

	int2 position = (int2)(get_global_id(0), get_global_id(1));

	int aexi;

	if (randFloat(&seedValue) < epsilon)
		// Exploratory action
		aexi = (int)(randFloat(&seedValue) * subActionCount);
	else
		aexi = (int)(round(read_imagef(actions, defaultSampler, position).x));
	
	write_imagef(actionsExploratory, position, (float4)(aexi));
}
