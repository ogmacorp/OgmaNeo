# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

""" Sin Prediction Demo """

import numpy as np
import matplotlib.pyplot as plt
import ogmaneo
import pkg_resources

print("OgmaNeo version: " + pkg_resources.get_distribution("ogmaneo").version)

includeErrors = True
graph_update_interval = 100

num_plots = 200
if includeErrors:
    num_plots = num_plots + 100


def updatePlot(plt, fig, i):
    plt.clf()
    fig.suptitle("OgmaNeo scalar prediction (one-step ahead)\nTraining steps: {}  Test steps: {}"
                 .format(train_steps, test_steps))

    test_start = iterations - test_steps + 1

    subplot_index = 10
    actual_plot = fig.add_subplot(num_plots + subplot_index + 1)
    subplot_index = subplot_index + 1
    predicted_plot = fig.add_subplot(num_plots + subplot_index + 1, sharex=actual_plot, sharey=actual_plot)
    subplot_index = subplot_index + 1
    if includeErrors:
        errors_plot = fig.add_subplot(num_plots + subplot_index + 1, sharex=actual_plot)

    actual_plot.set_xticks(np.arange(0, len(actual), 50.0))
    actual_plot.grid()
    actual_plot.plot(t, actual)
    actual_plot.axvline(test_start-1, 0, 1, color='r')
    actual_plot.set_ylabel('Input')
    actual_plot.set_ylim([-1.25, 1.25])

    predicted_plot.grid()
    predicted_plot.plot(t, values_neo)
    predicted_plot.axvline(test_start-1, 0, 1, color='r')
    predicted_plot.set_ylabel('OgmaNeo')
    predicted_plot.set_ylim([-1.25, 1.25])

    if includeErrors:
        errors_plot.grid()
        errors_plot.plot(t, errors_neo)
        errors_plot.axvline(test_start-1, 0, 1, color='r')
        errors_plot.set_ylabel('NRMSE')
        errors_plot.set_autoscaley_on(False)
        errors_plot.set_ylim([0, 1.0])

    plt.pause(0.001)


def matToVec(mat):
    return mat.flatten().astype(np.float32).tolist()


plt.ion()
fig = plt.figure(facecolor='white')
# Increase width of figure
fig_size = fig.get_size_inches()
fig.set_size_inches((fig_size[0] * 1.5, fig_size[1]), forward=True)
plt.show()

iterations = int(np.pi * 314)
test_steps = int((iterations * 25) / 100)
train_steps = iterations - test_steps

actual = np.zeros(iterations)
values_neo = np.zeros(iterations)
errors_neo = np.zeros(iterations)
t = range(0, iterations)

sequence = []

# Construct OgmaNeo network
res = ogmaneo.Resources()
res.create(ogmaneo.ComputeSystem._gpu)

# Generate sinus data
for i in range(0, iterations):
    v = np.sin(i * 0.3)
    actual[i] = v

    d = np.zeros((1, 1))
    d[0] = v
    sequence.append(d)

rmse_div = np.amax(actual) - np.amin(actual)

arch = ogmaneo.Architect()
arch.initialize(1234, res)

# Add an input layer that takes in the scalar value
inputField = ogmaneo.ValueField2D(ogmaneo.Vec2i(1, 1))
arch.addInputLayer(ogmaneo.Vec2i(1, 1))
arch.addHigherLayer(ogmaneo.Vec2i(96, 96), ogmaneo._distance)

for i in range(0, 1):
    # Add higher layers, using a chunk encoder
    layerParams = arch.addHigherLayer(ogmaneo.Vec2i(96, 96), ogmaneo._chunk)

neo_net = arch.generateHierarchy()

for i in range(0, train_steps):
    inputField.setValue(ogmaneo.Vec2i(0, 0), float(sequence[i]))

    inputVector = ogmaneo.vectorvf()
    inputVector.push_back(inputField)

    neo_net.activate(inputVector)
    neo_net.learn(inputVector)

    values_neo[i] = neo_net.getPredictions()[0].getValue(ogmaneo.Vec2i(0, 0))

    target_value = sequence[(i + 1) % len(sequence)]

    nrmse = np.sqrt(np.mean((values_neo[i] - target_value) ** 2)) / rmse_div
    errors_neo[i] = nrmse

    if (i % graph_update_interval) == 0 or i == train_steps - 1:
        updatePlot(plt, fig, i)
        print("Training: " + str(i) + "/" + str(train_steps))


for i in range(train_steps, iterations):
    inputField.setValue(ogmaneo.Vec2i(0, 0), float(values_neo[i-1]))

    inputVector = ogmaneo.vectorvf()
    inputVector.push_back(inputField)

    neo_net.activate(inputVector)
    neo_net.learn(inputVector)

    values_neo[i] = neo_net.getPredictions()[0].getValue(ogmaneo.Vec2i(0, 0))

    target_value = sequence[(i + 1) % len(sequence)]

    nrmse = np.sqrt(np.mean((values_neo[i] - target_value) ** 2)) / rmse_div
    errors_neo[i] = nrmse

    if (i % graph_update_interval) == 0 or i == iterations - 1:
        updatePlot(plt, fig, i)
        print("Prediction: " + str(i) + "/" + str(iterations))

plt.ioff()
plt.show()
#plt.savefig('figureX.png', facecolor=fig.get_facecolor(), transparent=True)