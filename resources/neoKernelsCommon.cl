// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ----------------------------------------- Samplers -----------------------------------------

constant sampler_t defaultSampler = CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP |
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
    return (int2)((position.x + 0.5f) * toScalars.x + 0.5f, (position.y + 0.5f) * toScalars.y + 0.5f);
}

int2 projectf(float2 position, float2 toScalars) {
    return (int2)((position.x + 0.5f) * toScalars.x + 0.5f, (position.y + 0.5f) * toScalars.y + 0.5f);
}

// Initialize a random uniform 2D image
void kernel randomUniform2D(write_only image2d_t values, uint2 seed, float4 lowerBounds, float4 upperBounds, float4 mask, float4 fillConstants) {
    uint2 seedValue = seed + (uint2)(get_global_id(0) * 29 + 12, get_global_id(1) * 16 + 23) * 36;

    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float4 randVal = (float4)(randFloat(&seedValue) * (upperBounds.x - lowerBounds.x) + lowerBounds.x,
		randFloat(&seedValue) * (upperBounds.y - lowerBounds.y) + lowerBounds.y,
		randFloat(&seedValue) * (upperBounds.z - lowerBounds.z) + lowerBounds.z,
		randFloat(&seedValue) * (upperBounds.w - lowerBounds.w) + lowerBounds.w);

    write_imagef(values, position, mask * randVal + (1.0f - mask) * fillConstants);
}

// Initialize a random uniform 3D image
void kernel randomUniform3D(write_only image3d_t values, uint2 seed, float4 lowerBounds, float4 upperBounds, float4 mask, float4 fillConstants) {
    uint2 seedValue = seed + (uint2)(get_global_id(0) * 12 + 76 + get_global_id(2) * 3, get_global_id(1) * 21 + 42 + get_global_id(2) * 7) * 12;

    int3 position = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));

    float4 randVal = (float4)(randFloat(&seedValue) * (upperBounds.x - lowerBounds.x) + lowerBounds.x,
		randFloat(&seedValue) * (upperBounds.y - lowerBounds.y) + lowerBounds.y,
		randFloat(&seedValue) * (upperBounds.z - lowerBounds.z) + lowerBounds.z,
		randFloat(&seedValue) * (upperBounds.w - lowerBounds.w) + lowerBounds.w);

    write_imagef(values, (int4)(position, 0), mask * randVal + (1.0f - mask) * fillConstants);
}