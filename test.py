import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
from stable_baselines3 import SAC

# env
env = gym.make('Walker2DMuJoCoEnv-v0')
env.render(mode='human') # call this before env.reset, if you want a window showing the environment
env.reset()  # should return a state vector if everything worked

# model
model = SAC.load("model")

# enjoy
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode='human')
    if done:
      obs = env.reset()