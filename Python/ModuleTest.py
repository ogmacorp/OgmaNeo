# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import ogmaneo

import pkg_resources
print("OgmaNeo version: " + pkg_resources.get_distribution("ogmaneo").version)

res = ogmaneo.Resources()
res.create(ogmaneo.ComputeSystem._cpu, 0, 0)

arch = ogmaneo.Architect()
arch.initialize(1234, res)

w, h = 16, 16

# Add an input layer, and adjust some parameters
inputParams = arch.addInputLayer(ogmaneo.Vec2i(16, 16))
inputParams.setValue("in_p_alpha", 0.02)
inputParams.setValue("in_p_radius", 16)

# Add a coupld of higher layers, and adjust some parameters
for x in range(0, 2):
    layerParams = arch.addHigherLayer(ogmaneo.Vec2i(256, 256), ogmaneo._chunk)
    layerParams.setValue("p_alpha", 0.08)
    layerParams.setValue("p_beta", 0.16)
    layerParams.setValue("p_radius", 12)

# Generate the hierarchy
hierarchy = arch.generateHierarchy()

print("Done")
