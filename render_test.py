#!/usr/bin/env python

import gym
import numpy as np
from atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

def wrap_train(env):
    from atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, episode_life = False, clip_rewards=False)
    env = FrameStack(env, 4)
    return env

env = gym.make("PongNoFrameskip-v4")
env = ScaledFloatFrame(wrap_dqn(env))
# env = wrap_train(env)
obs = env.reset()

print env.observation_space
print env.action_space

print len(obs), len(obs[0]), len(obs[0][0]) 
action = env.action_space.sample()
print action

# print len(observation)
# for _ in range(1000):
#     # env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     print action
#     if done:
#         env.reset()