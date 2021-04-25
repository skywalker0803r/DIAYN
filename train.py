from env_wrapper import DIAYN_Skill_Wrapper
import gym
import pybulletgym
from stable_baselines3 import SAC

total_timesteps = 10000
num_skills = 100
env = gym.make('Walker2DMuJoCoEnv-v0')
env = DIAYN_Skill_Wrapper(env,num_skills=num_skills)
agent = SAC("MlpPolicy", env, verbose=1,tensorboard_log="./tensorboard").learn(total_timesteps=total_timesteps)
agent.save("sac_Walker2DMuJoCoEnv-v0")