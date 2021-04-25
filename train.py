import gym
import pybulletgym
import time
import numpy as np
from model import SAC,Discriminator
from utils import sample_skill_and_initialize_state,sample_action_from_skill,update_policy,update_discriminator


# config
num_skills = 10

# environment
env = gym.make('Walker2DMuJoCoEnv-v0')
env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# policy
policy = SAC(state_dim,num_skills,action_dim)

# discriminator
discriminator = Discriminator(state_dim,num_skills)

# DIAYN 
def main():
    converged = False
    while not converged:
        state,skill = sample_skill_and_initialize_state(env,num_skills)
        action = sample_action_from_skill(policy,state,skill)
        next_state ,_ ,done ,_ = env.step(action)
        predicted_probability = discriminator.predict(next_state)
        reward = np.log(predicted_probability[skill]+ 1e-8) - np.log(1/num_skills)
        update_policy(policy,skill,state,action,reward,next_state,done,num_skills)
        update_discriminator(discriminator,next_state,predicted_probability,skill)

if __name__ == '__main__':
    main()