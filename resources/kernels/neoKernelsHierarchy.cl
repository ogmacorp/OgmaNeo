// ----------------------------------------------------------------------------
//  OgmaNeo
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the OGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

// ------------------------------------------ Feature Hierarchy ------------------------------------------

void kernel fhPool(read_only image2d_t states, read_only image2d_t outputsBack, write_only image2d_t outputsFront, float scale) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float state = read_imagef(states, defaultSampler, position).x;

    float outputPrev = read_imagef(outputsBack, defaultSampler, position).x;

    write_imagef(outputsFront, position, (float4)(fmax(outputPrev, state), 0.0f, 0.0f, 0.0f));
}

void kernel fhPredError(read_only image2d_t states, read_only image2d_t predictionsPrev, write_only image2d_t errors) {
    int2 position = (int2)(get_global_id(0), get_global_id(1));

    float state = read_imagef(states, defaultSampler, position).x;

    float predictionPrev = read_imagef(predictionsPrev, defaultSampler, position).x;

    //write_imagef(errors, position, (float4)(state - predictionPrev, 0.0f, 0.0f, 0.0f));
    //write_imagef(errors, position, (float4)(state, 0.0f, 0.0f, 0.0f));
	write_imagef(errors, position, (float4)(state * (1.0f - predictionPrev) + (1.0f - state) * predictionPrev, 0.0f, 0.0f, 0.0f));
}