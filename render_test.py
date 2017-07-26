#!/usr/bin/env python

import gym

def wrap_train(env):
    from atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, episode_life = False, clip_rewards=False)
    env = FrameStack(env, 4)
    return env

env = gym.make("Reacher-v1")
env = wrap_train(env)
obs = env.reset()

print env.observation_space
print env.action_space

print len(obs), len(obs[0]), len(obs[0][0]) 
# # print len(observation)
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     if done:
#         env.reset()