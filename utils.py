import numpy as np

def sample_skill_and_initialize_state(env,num_skills):
    state = env.reset()
    skill = np.random.choice([*range(num_skills)])
    return state,skill

def sample_action_from_skill(policy,state,skill):
    action = policy.select_action(state,skill)
    return action

def update_policy(policy,skill,state,action,reward,next_state,done,num_skills):
    skill_one_hot = [ 0 for i in range(num_skills)]
    skill_one_hot[skill] = 1.0
    skill_one_hot = np.array(skill_one_hot)
    policy.replay_buffer.store(state,action,reward,next_state,done,skill_one_hot)
    if len(policy.replay_buffer) > 64:
        policy.update(update_epochs = 1)
        


def update_discriminator(discriminator,next_state,predicted_probability,skill):
    return 0
    