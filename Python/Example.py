# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import numpy as np
import os.path
import ogmaneo

import pkg_resources
print("OgmaNeo version: " + pkg_resources.get_distribution("ogmaneo").version)

serializationEnabled = False

numSimSteps = 200

res = ogmaneo.Resources()
res.create(ogmaneo.ComputeSystem._gpu)

arch = ogmaneo.Architect()
arch.initialize(1234, res)

w, h = 4, 4

inputParams = arch.addInputLayer(ogmaneo.Vec2i(w, h))
inputParams.setValue("in_p_alpha", 0.02)            # Learning rate.
inputParams.setValue("in_p_radius", 8)              # Input field radius (onto first hidden layer).

for x in range(0, 2):
    # Add higher layers, that use delay encoders
    layerParams = arch.addHigherLayer(ogmaneo.Vec2i(32, 32), ogmaneo._chunk)
    layerParams.setValue("sfc_numSamples", 2)

hierarchy = arch.generateHierarchy()

inputField = ogmaneo.ValueField2D(ogmaneo.Vec2i(w, h))

for y in range(h):
    for x in range(w):
        inputField.setValue(ogmaneo.Vec2i(x,y), (y*w)+x)

if (serializationEnabled and os.path.exists("example.opr")):
    print("Loading hierarchy from example.opr")
    hierarchy.load(res.getComputeSystem(), "example.opr")

inputVector = ogmaneo.vectorvf()
inputVector.push_back(inputField)

print("Stepping the hierarchy...")
for i in range(0, numSimSteps):
    hierarchy.activate(inputVector)
    hierarchy.learn(inputVector)

prediction = hierarchy.getPredictions()[0]

outStr = ''
size = inputField.getSize()
for y in range(0, size.y):
    for x in range(0, size.x):
        outStr += '{0:.2f}'.format(inputField.getValue(ogmaneo.Vec2i(x, y))) + ' '
print("Input     : " + outStr)

outStr = ''
size = prediction.getSize()
for y in range(0, size.y):
    for x in range(0, size.x):
        outStr += '{0:.2f}'.format(prediction.getValue(ogmaneo.Vec2i(x, y))) + ' '
print("Prediction: " + outStr)

if (serializationEnabled):
    print("Saving hierachy to example.opr")
    hierarchy.save(res.getComputeSystem(), "example.opr")

print("Done")
