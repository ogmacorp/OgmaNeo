# -----------------------------------------------------------------------------
# Ogma Toolkit (OTK)
# Copyright (c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
# -----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

"""
Mackey-Glass sequence prediction comparison, using OTK Neo

Mackey-Glass function parameters are setup to match the paper;
a=0.2, b=0.1, c=10, tau=17, with step size 0.1, and initial condition of 1.2 for t<0
Felix Gers, Douglas Eck, Jurgen Schmidhuber - Applying LSTM to Time Series
Predictable Through Time-Window Approaches. dl.acm.org/citation.cfm?id=870454

"""

import ogmaneo
import Oger
import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

# oger_samples_file and oger_output_file ultimately determine the num_samples
# changes to num_samples thus requires those files to be rebuilt by the Oger network
num_samples = 2000

seed = 1337


def matToVec(mat):
    return mat.flatten().astype(np.float32).tolist()


def updatePlots(plt, fig, current_step, oger_values):
    plt.clf()
    fig.suptitle("Training steps: {}  Test steps: {}".format(train_steps, test_steps))

    actual_plot = fig.add_subplot(411)
    oger_plot = fig.add_subplot(412, sharex=actual_plot, sharey=actual_plot)
    predicted_plot = fig.add_subplot(413, sharex=oger_plot, sharey=oger_plot)
    errors_plot = fig.add_subplot(414, sharex=oger_plot)

    test_start = num_samples - test_steps + 1

    actual_plot.plot(t, actual)
    actual_plot.set_ylabel('Input')
    actual_plot.axvline(test_start, 0, 1, color='r')
    actual_plot.axvline(current_step, 0, 0.5, color='r')

    oger_plot.plot(t, oger_values)
    oger_plot.set_ylabel('Oger')

    predicted_plot.plot(t, values)
    predicted_plot.set_ylabel('Ogma Neo')
    predicted_plot.axvline(current_step, 0.5, 1, color='r')

    errors_plot.plot(t, errors)
    errors_plot.set_ylabel('NRMSE')
    errors_plot.set_ylim([0, 1.0])
    errors_plot.axvline(test_start, 0, 1, color='r')

    plt.pause(0.001)


def trainOgerNetwork():
    """
    Source: http://organic.elis.ugent.be/node/364
  
    Train a standard reservoir + readout to do autonomous signal generation.
    This is done by training the readout to perform one - step - ahead prediction
    on a teacher signal. After training, we then feed the reservoir its own
    prediction, and it then autonomously generates the signal for a certain time.
  
    MDP only supports feed-forward flows, so how can we solve this? Oger provides
    a FeedbackFlow which is suited for precisely this case. The training and execute
    functions are overridden from the original Flow class. A FeedbackFlow is suitable
    for signal generation tasks both without and with external additional inputs.
    The example below uses no external inputs, so the flow generates its own output.
    After training, for every timestep, the output of the Flow is fed back as input
    again, such that it can autonomously generate the desired signal.
  
    The train function of a FeedbackFlow will take the timeseries given as training
    argument and internally construct a new target signal which is shifted one timestep
    into the past, such that the flow is trained to do one step ahead prediction.
    During execution (i.e. by calling the execute function), the flow will first be
    teacher-forced using the first portion provided input signal, and starting from
    the timestep freerun_steps from the end of the signal, the flow is run in freerun
    mode, i.e. being fed back its own prediction.
    """

    # Preparing the dataset. In this case, there is no target signal, so the dataset
    # consists of only a single time-series. We construct the target signal as the
    # time-series shifted over one timestep to the left (we want one-step ahead prediction).

    training_sample_length = num_samples
    n_training_samples = num_samples / 1000
    test_sample_length = num_samples

    train_signals = Oger.datasets.mackey_glass(
        sample_len=training_sample_length,
        tau=17, seed=seed,
        n_samples=n_training_samples)
    test_signals = Oger.datasets.mackey_glass(
        sample_len=test_sample_length,
        tau=17, seed=seed,
        n_samples=1)

    freerun_steps = (num_samples * 10) / 100

    # Create a reservoir with a little bit of leak rate and a readout node and combine
    # these into a FeedbackFlow.Notice how during creation of the FeedbackFlow, we pass
    # it the keyword argument of freerun_steps.This determines how many timesteps the
    # FeedbackFlow will run in 'freerun mode'.

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=400, leak_rate=0.4, input_scaling=.05, bias_scaling=.2,
                                              reset_states=False)
    readout = Oger.nodes.RidgeRegressionNode()
    Oger.utils.enable_washout(Oger.nodes.RidgeRegressionNode, 500)

    flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=freerun_steps)

    # Instantiate an Optimizer for finding the best regularization constant for the readout.
    # This is necessary to achieve a suitable robustness and stability of the generated model.
    gridsearch_parameters = {readout: {'ridge_param': 10 ** scipy.arange(-4, 0, .3)}}

    loss_function = Oger.utils.timeslice(
        range(training_sample_length - freerun_steps, training_sample_length),
        Oger.utils.nrmse)

    opt = Oger.evaluation.Optimizer(gridsearch_parameters, loss_function)

    # Run a gridsearch of the ridge parameter, providing the set of training signals as
    # dataset, and using leave one out cross-validation.
    opt.grid_search([[], train_signals], flow, cross_validate_function=Oger.evaluation.leave_one_out)

    # Retrieve the optimal flow, train it on the full training set and evaluate
    # its performance on the unseen test signals.
    opt_flow = opt.get_optimal_flow(verbose=True)
    opt_flow.train([[], train_signals])

    # print 'Freerun on test_signals signal with the optimal flow...'
    freerun_output = opt_flow.execute(test_signals[0][0])

    plot_oger = False
    if plot_oger:
        # Plot an overlay of the signal generated by the reservoir, and the actual target signal:
        plt.plot(scipy.concatenate((test_signals[0][0][-2 * freerun_steps:])))
        plt.plot(scipy.concatenate((freerun_output[-2 * freerun_steps:])))
        plt.xlabel('Timestep')
        plt.legend(['Target signal', 'Predicted signal'])
        plt.axvline(plt.xlim()[1] - freerun_steps + 1, plt.ylim()[0], plt.ylim()[1], color='r')
        # print opt_flow[1].ridge_param
        plt.show()

    return test_signals, freerun_output


oger_samples_file = "oger_samples.npy"
oger_output_file = "oger_output.npy"

if os.path.isfile(oger_samples_file) and os.path.isfile(oger_output_file):
    samples = np.load(oger_samples_file)
    oger_output = np.load(oger_output_file)
    num_samples = len(samples)
    total_iterations = num_samples
    test_steps = (num_samples * 10) / 100
    train_steps = total_iterations - test_steps
else:
    samples, oger_output = trainOgerNetwork()
    samples = np.ravel(samples)[0:num_samples]  # Flatten array
    np.save(oger_samples_file, samples)
    np.save(oger_output_file, oger_output)

# Compare samples and Oger freerun results
# t = range(0, len(samples))
# plt.plot(t, samples, t, oger_output)
# plt.show()

total_iterations = num_samples
test_steps = int((num_samples * 10) / 100)
train_steps = total_iterations - test_steps

actual = np.zeros(total_iterations)
values = np.zeros(total_iterations)
errors = np.zeros(total_iterations)
t = range(0, total_iterations)

plt.ion()
fig = plt.figure(facecolor='white')
plt.show()

# Trim sequence
sequence = []

# Grab some samples
for i in range(0, total_iterations):
    v = samples[i % len(samples)]  # np.sin(i * 0.01)
    d = np.zeros((1, 1))
    d[0] = v
    actual[i] = v
    sequence.append(d)

rmse_div = np.amax(actual) - np.amin(actual)

seed = 12312
encoder_size = 64
neo_layer_size = 1024
decoder_multiplier = 100.0

encoder_width = int(np.sqrt(encoder_size))
layer_width = int(np.sqrt(neo_layer_size))

# Create OgmaNeo Hierarchy
res = ogmaneo.Resources()
res.create(ogmaneo.ComputeSystem._gpu)

arch = ogmaneo.Architect()
arch.initialize(seed, res)

# First layer scalarEncoder for scalars, subsequent layers in Neo
scalarEncoder = ogmaneo.ScalarEncoder()
scalarEncoder.createRandom(1, encoder_size, -0.01, 0.01, seed)

inputField = ogmaneo.ValueField2D(ogmaneo.Vec2i(encoder_width, encoder_width))

inputParams = arch.addInputLayer(ogmaneo.Vec2i(encoder_width, encoder_width))
inputParams.setValue("in_p_alpha", 0.02)
inputParams.setValue("in_p_radius", encoder_width)

for x in range(0, 3):
    layerParams = arch.addHigherLayer(ogmaneo.Vec2i(layer_width, layer_width), ogmaneo._chunk)

for x in range(0, 3):
    layerParams = arch.addHigherLayer(ogmaneo.Vec2i(layer_width, layer_width), ogmaneo._stdp)

hierarchy = arch.generateHierarchy()

prediction_sdr = np.zeros(encoder_size)
prediction = None

for i in range(0, train_steps):
    scalarEncoder.encode(matToVec(sequence[i]), 0.5, 0.0, 0.0)
    encodedInput = scalarEncoder.getEncoderOutputs()
    for j in range(encoder_size):
        inputField.setValue(ogmaneo.Vec2i(j % 8, j // 8), encodedInput[j])

    inputVector = ogmaneo.vectorvf()
    inputVector.push_back(inputField)

    hierarchy.simStep(inputVector, True)

    prediction = hierarchy.getPredictions()[0]

    for j in range(encoder_size):
        prediction_sdr[j] = prediction.getValue(ogmaneo.Vec2i(j % 8, j // 8))

    scalarEncoder.decode(prediction_sdr)

    values[i] = scalarEncoder.getDecoderOutputs()[0] * decoder_multiplier

    targ = sequence[(i + 1) % len(sequence)]
    nrmse = np.sqrt(np.mean((values[i] - targ) ** 2)) / rmse_div
    errors[i] = nrmse

    if (i % 100) == 0 or i == total_iterations - 1:
        updatePlots(plt, fig, i, oger_output)
        print("Train: " + str(i) + "/" + str(train_steps) + " NRMSE: " + str(nrmse))

for i in range(train_steps, total_iterations):
    scalarEncoder.encode(matToVec(sequence[i]), 0.5, 0.0, 0.0)
    encodedInput = scalarEncoder.getEncoderOutputs()

    for j in range(encoder_size):
        inputField.setValue(ogmaneo.Vec2i(j % 8, j // 8), encodedInput[j])

    inputVector = ogmaneo.vectorvf()
    inputVector.push_back(inputField)

    hierarchy.simStep(inputVector, True)

    prediction = hierarchy.getPredictions()[0]

    for j in range(encoder_size):
        prediction_sdr[j] = prediction.getValue(ogmaneo.Vec2i(j % 8, j // 8))

    scalarEncoder.decode(prediction_sdr)

    values[i] = scalarEncoder.getDecoderOutputs()[0] * decoder_multiplier

    targ = sequence[(i + 1) % len(sequence)]
    nrmse = np.sqrt(np.mean((values[i] - targ) ** 2)) / rmse_div
    errors[i] = nrmse

    if (i % 100) == 0 or i == total_iterations - 1:
        updatePlots(plt, fig, i, oger_output)
        print("Prediction: " + str(i) + "/" + str(total_iterations) + " NRMSE: " + str(nrmse))

plt.ioff()
plt.show()
