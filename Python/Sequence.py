# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of POgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import ogmaneo
import numpy as np
from copy import copy

sequence = [[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]]

def arrToList(mat):
    return ogmaneo.vectorf(mat.astype(np.float32).tolist())

trainIter = 1500 # Number of entries to iterate over

res = ogmaneo.Resources()
res.create(ogmaneo.ComputeSystem._gpu)

seed = 23423

arch = ogmaneo.Architect()
arch.initialize(seed, res)


# Hierarchy parameters
rootSize = int(np.ceil(np.sqrt(len(sequence[0])))) # Fit into a square
inputWidth = rootSize
inputHeight = rootSize
totalInputSize = inputWidth * inputHeight
leftovers = totalInputSize - len(sequence[0])

inputParams = arch.addInputLayer(ogmaneo.Vec2i(inputWidth, inputHeight))
inputParams.setValue("in_p_alpha", 0.02)
inputParams.setValue("in_p_radius", totalInputSize)

for x in range(0, 3):
    layerParams = arch.addHigherLayer(ogmaneo.Vec2i(32, 32), ogmaneo._chunk)
    layerParams.setValue("sfc_chunkSize", ogmaneo.Vec2i(4, 4))

h = arch.generateHierarchy()

inputField = ogmaneo.ValueField2D(ogmaneo.Vec2i(inputWidth, inputHeight))

pos = ogmaneo.Vec2i()

for i in range(0, trainIter):
    itemLen = len(sequence[i % len(sequence)])
    for j in range(0, itemLen):
        pos.x = j % inputWidth
        pos.y = j // inputWidth
        inputField.setValue(pos, sequence[i % len(sequence)][j])

    for j in range(itemLen, itemLen+leftovers):
        pos.x = j % inputWidth
        pos.y = j // inputWidth
        inputField.setValue(pos, 0.0)

    inputVector = ogmaneo.vectorvf()
    inputVector.push_back(inputField)

    h.activate(inputVector)
    h.learn(inputVector)

    match = True

    prediction = h.getPredictions()[0]
    for j in range(0, itemLen):
        pos.x = j % inputWidth
        pos.y = j // inputWidth
        print('%f == %f' % (prediction.getValue(pos), sequence[(i + 1) % len(sequence)][j]))
        if (prediction.getValue(pos) > 0.5) != (sequence[(i + 1) % len(sequence)][j] > 0.5):
            match = False

    print(match)
