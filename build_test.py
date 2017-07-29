#!/usr/bin/env python

import gym
import numpy as np
from atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
from models import *

def wrap_train(env):
    from atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, episode_life = False, clip_rewards=False)
    env = FrameStack(env, 4)
    return env

# env = gym.make("PongNoFrameskip-v4")
# env = ScaledFloatFrame(wrap_dqn(env))
# # env = wrap_train(env)
# obs = env.reset()

# print env.observation_space
# print env.action_space

# print len(obs), len(obs[0]), len(obs[0][0]) 
# action = env.action_space.sample()
# print action

# print len(observation)
# for _ in range(1000):
#     # env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     print action
#     if done:
#         env.reset()


def main():
    env = gym.make("PongNoFrameskip-v4")
    # Remove Scaled Float Frame wrapper, re-use if needed.
    from atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
    env = ScaledFloatFrame(wrap_dqn(env))
    model = cnn_to_dist(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        num_atoms = 50,
        dueling=True
    )
    # act = learn(
    #     env,
    #     q_func=model,
    #     lr=1e-4,
    #     max_timesteps=2000000,
    #     buffer_size=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.01,
    #     train_freq=4,
    #     learning_starts=10000,
    #     target_network_update_freq=1000,
    #     gamma=0.99,
    #     prioritized_replay=False
    # )
    # act.save("pong_model.pkl")
    # env.close()


if __name__ == '__main__':
    main()