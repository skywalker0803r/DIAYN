import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from replay_buffer import ExperienceReplayBuffer

class ActorNetwork(nn.Module):
    def __init__(self,input_dim ,output_dim ,hidden_dim=256):
        super(ActorNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim ,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim ,hidden_dim),
            nn.ReLU(),
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim ,output_dim),
            nn.Sigmoid(),
        )
    
    def forward(self,state,skill):
        latent = self.feature_extractor(torch.cat((state ,skill),dim=-1))
        action = self.output_head(latent)
        return action

class CriticNetwork(nn.Module):
    def __init__(self,input_dim ,output_dim ,hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim ,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim ,hidden_dim),
            nn.ReLU(),
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim ,output_dim),
        )
    
    def forward(self,state,skill):
        latent = self.feature_extractor(torch.cat((state ,skill),dim=-1))
        Q_value = self.output_head(latent)
        return Q_value

class SAC(object):
    def __init__(self,state_dim,skill_dim,output_dim,hidden_dim=256,buffer_size=9999):
        super(SAC, self).__init__()
        
        # dim
        self.state_dim = state_dim
        self.skill_dim = skill_dim
        self.output_dim = output_dim
        
        # actor
        self.actor_network = ActorNetwork(state_dim + skill_dim ,output_dim ,hidden_dim)
        
        # critic_1 and critic1_target 
        self.critic_network_1 = CriticNetwork(state_dim + skill_dim ,output_dim ,hidden_dim)
        self.critic_network_target_1 = CriticNetwork(state_dim + skill_dim ,output_dim ,hidden_dim)
        self.hard_update(self.critic_network_1,self.critic_network_target_1)
        
        # critic_2 and critic2_target
        self.critic_network_2 = CriticNetwork(state_dim + skill_dim ,output_dim ,hidden_dim)
        self.critic_network_target_2 = CriticNetwork(state_dim + skill_dim ,output_dim ,hidden_dim)
        self.hard_update(self.critic_network_2,self.critic_network_target_2)
        
        # optimizer
        self.actor_optimizer = Adam(self.actor_network.parameters())
        self.critic_optimizer_1 = Adam(self.critic_network_1.parameters())
        self.critic_optimizer_2 = Adam(self.critic_network_2.parameters())
        
        # replay_buffer
        self.replay_buffer = ExperienceReplayBuffer(size=buffer_size)
    
    
    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def select_action(self,state,skill_idx):
        state = torch.FloatTensor(state)
        skill = torch.zeros(self.skill_dim)
        skill[skill_idx] = float(1.0)
        action = self.actor_network(state,skill)
        return action.detach().numpy()
    
    def update(self,update_epochs):
        for j in range(update_epochs):
            minibatch = self.replay_buffer.sample(batch_size=64)
            Q_target = self.compute_Q_target(minibatch)
            #update_Q_function(Q_target)
            #update_policy
            print("=================================")
            print(len(minibatch))
            print(minibatch[0][0].shape)#state
            print(minibatch[0][1].shape)#action
            print(minibatch[0][2])#reward
            print(minibatch[0][3].shape)#next_state
            print(minibatch[0][4])#done
            print(minibatch[0][5])#skill
            print("=================================")
            raise '!!!!!!!!!!!!!!'
    
    def compute_Q_target(self,minibatch,gamma=0.99):
        rewards = [ trajectory[2] for trajectory in minibatch]
        dones = [ trajectory[4] for trajectory in minibatch]
        next_states = [ trajectory[3] for trajectory in minibatch]
        skill_one_hots =  [ trajectory[5] for trajectory in minibatch]
        next_actions = self.actor_network(torch.FloatTensor(next_states),torch.FloatTensor(skill_one_hots))
        print(next_actions)
        raise '456'
        
        print(torch.FloatTensor(next_states).shape)
        print(torch.FloatTensor(skill_one_hots).shape)
        Q1s = self.critic_network_target_1(torch.FloatTensor(next_states),torch.FloatTensor(skill_one_hots))
        Q2s = self.critic_network_target_2(torch.FloatTensor(next_states),torch.FloatTensor(skill_one_hots))
        
        print(Q1s.shape)
        print(Q2s.shape)
        print(Q1s.min(dim=-1)[0])
        raise '123'
        

# DiscriminatorNetwork and Discriminator
class DiscriminatorNetwork(nn.Module):
    def __init__(self,input_dim ,output_dim ,hidden_dim=256):
        super(DiscriminatorNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim ,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim ,hidden_dim),
            nn.ReLU(),
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim ,output_dim),
            nn.Softmax(),
        )
    
    def forward(self,state):
        latent = self.feature_extractor(state)
        output = self.output_head(latent)
        return output

class Discriminator(object):
    def __init__(self,state_dim,skill_dim,hidden_dim=256):
        super(Discriminator,self).__init__()
        
        self.state_dim = state_dim
        self.skill_dim = skill_dim
        
        self.discriminator_network = DiscriminatorNetwork(state_dim,skill_dim)
        self.optimizer = Adam(self.discriminator_network.parameters(),lr=1e-3)
    
    def predict(self,state):
        state = torch.FloatTensor(state)
        output = self.discriminator_network(state)
        return output.detach().numpy()
        