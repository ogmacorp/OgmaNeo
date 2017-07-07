# ----------------------------------------------------------------------------
#  PyOgmaNeo
#  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyOgmaNeo is licensed to you under the terms described
#  in the PYOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import gym
import ogmaneo

res = 32

def matToVec(mat):
    return np.asarray(mat.flatten()).astype(np.float32).tolist()

class NeoAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

        self._csi = ogmaneo.ComputeSystemInterface()
        self._cs = self._csi.create(ogmaneo.ComputeSystem._gpu)

        self._prog = ogmaneo.ComputeProgramInterface()
        self._prog.loadMainKernel(self._csi)

        self._actions = np.zeros((1, 1))
        
        layerDescs = [ ogmaneo.LayerDescs(32, 32), ogmaneo.LayerDescs(24, 24), ogmaneo.LayerDescs(16, 16), ogmaneo.LayerDescs(8, 8) ]

        for l in layerDescs:
            l._spBiasAlpha = 0.0
            l._spActiveRatio = 0.06

            l._spFeedForwardWeightAlpha = 0.001

            l._epsilon = 0.005
            l._qAlpha = 0.1
            l._qGamma = 0.96
            l._qLambda = 0.94

        self._agent = ogmaneo.Agent(self._csi(), self._prog(), 8, 8, 1, 1, 3, 3, 10, layerDescs, -0.01, 0.01, 1234)

        self._se = ogmaneo.ScalarEncoder()
        self._se.createRandom(observation_space.shape[0], 64, -1.0, 1.0, 4321)

    def act(self, observation, reward, done):
        self._se.encode(matToVec(observation), 0.1, 0.0, 0.0)

        self._agent.simStep(-float(done), self._se.getEncoderOutputs(), True)

        # Transform into one-hot vector
        self._actions = np.matrix(self._agent.getAction()).T / float(3 * 3 - 1)

        act = 0

        if self._actions[0] > 0.5:
            act = 1
        else:
            act = 0

        return act

if __name__ == '__main__':
    # You can optionally set up the logger. Also fine to set the level
    # to logging.DEBUG or logging.WARN if you want to change the
    # amount of outut.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v0')
    agent = NeoAgent(env.action_space, env.observation_space)

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env.monitor.start(outdir, force=True)#video_callable=lambda i : False

    episode_count = 800
    max_steps = 100000
    reward = 0
    totalReward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            totalReward += reward
            #env.render()
            if done:
                print("Total reward: " + str(totalReward))
                totalReward = 0
                break

    # Dump result info to disk
    env.monitor.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    #logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    #gym.upload(outdir, algorithm_id='random')