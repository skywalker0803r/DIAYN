from env_wrapper import DIAYN_Skill_Wrapper
import gym
import pybulletgym
from stable_baselines3 import SAC
import time

class DIAYN_Skill_Wrapper_test(DIAYN_Skill_Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.skill = 0
        return self.observation(observation)

model = SAC.load("sac_Walker2DMuJoCoEnv-v0.zip")
env = gym.make('Walker2DMuJoCoEnv-v0')
env = DIAYN_Skill_Wrapper_test(env,num_skills=2)
env.render(mode='human')
obs = env.reset()

while True:
    time.sleep(1/60)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs[-1])
    env.render(mode='human')
    if done:
      obs = env.reset()