import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 1)
        

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
                
        
        mean=self.linear3(output)
        sigma=self.linear3(output)
        sigma=F.softplus(sigma)
        distribution = torch.distributions.normal.Normal(mean,sigma)
        return distribution

        
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 1)
        
    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value



class RandomAgent():
    def __init__(self):

        """Init a new agent.
        """
        self.gamma=0.99
        self.lrA=0.001
        self.lrC=0.0001
        self.epsilon=0.1
        self.reset_step=500
        self.state_size=2
        self.action_size =1
        self.scaler=None
        self.featurizer=None
        self.first_run_reset=True
        self.Transition=None
        
		
        self.episode = list()
        self.episode=0
        self.current_state=None
        self.current_action=None
        self.current_reward=None
        self.next_state=None
        self.xrange=None
        self.limit_act_bol=False
        self.win_count=0
        self.iteration=0
        self.observation_list=list()
        self.reward_list=list()  
        self.done_MC=0
        self.torch=torch
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropy = 0
        
		
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
        self.Actor = Actor(self.state_size, self.action_size).to(self.device)
        self.Critic = Critic(self.state_size, self.action_size).to(self.device)   
        self.MC_Reset=1        
        
       
        self.optimizerA = optim.Adam(self.Actor.parameters(),lr=self.lrA)
        self.optimizerC = optim.Adam(self.Critic.parameters(),lr=self.lrC)

    
    
    
    def reset(self, x_range):
        
        self.episode=self.episode+1
        self.iteration_episode=0
        self.iteration=(self.episode-1)*400
        


                    
        if self.first_run_reset==True:
                        
            self.xrange=x_range
            self.get_observations()
            
            self.first_run_reset=False


    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        
        
        self.iteration_episode=self.iteration_episode+1
        
        if self.iteration_episode>1:
            self.learn(observation)

        self.current_state = torch.FloatTensor(self.featurize_state(observation)).to(self.device)
        self.distribution = self.Actor.forward(self.current_state )
        self.value = self.Critic.forward(self.current_state )

 
               
       
        self.action = self.torch.clamp(self.distribution.sample(),-10,10)
        
        
        if np.random.randint(0,10000)/10000<self.epsilon:
            self.action= np.array(np.random.randint(-10000,10000)/1000)
            self.action=self.torch.tensor(self.action)

        self.log_prob = self.distribution.log_prob(self.action).unsqueeze(0)
        self.entropy += self.distribution.entropy().mean()

        
        
        return float(self.action.item())

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """

        if reward > 0 :
            print("win")                
            self.done=1  
            self.done_MC=1

        else:
            self.done=0

        self.current_state=observation
        self.current_action=action
        self.current_reward=reward
        self.reward_list.append(reward.item())
            
        self.log_probs.append(self.log_prob)
        self.values.append(self.value)
        self.rewards.append(self.torch.tensor([self.current_reward], dtype=self.torch.float, device=self.device))
        self.masks.append(self.torch.tensor([1-self.done], dtype=self.torch.float, device=self.device))



    #for neural network to learn we will use this learn function

    def learn(self,observation):
		
            
        if self.MC_Reset%self.reset_step==0 or self.done_MC==1:
                     
            self.next_state=observation                          
            self.next_state=self.featurize_state(self.next_state)
                     
        
            self.next_state = torch.FloatTensor(self.next_state).to(self.device)    
            self.next_value = self.Critic.forward(self.next_state)                                                    
            self.R = self.next_value
                      
            self.returns = []
            for step in reversed(range(len(self.rewards))):
                self.R = self.rewards[step] + self.gamma * self.R* self.masks[step]
                self.returns.insert(0, self.R)
        
            self.returns = torch.cat(self.returns).detach()
            self.log_probs = torch.cat(self.log_probs)
            self.values = torch.cat(self.values)
        
            self.advantage = self.returns - self.value                                            
        
            self.actor_loss = -(self.log_probs * self.advantage.detach()).mean()
            self.critic_loss = self.advantage.pow(2).mean()
                      
            self.optimizerA.zero_grad()
            self.optimizerC.zero_grad()
            self.actor_loss.backward()
            self.critic_loss.backward()
            self.optimizerA.step()
            self.optimizerC.step()
                                                     
        
            if self.done_MC==1:
                self.done_MC=0
        
            self.MC_Reset=1
            self.log_probs = []
            self.values = []
            self.rewards = []
            self.masks = []
                     
        
        else:
        
            self.MC_Reset=self.MC_Reset+1
      
             

    def get_observations(self):        
        sample_x=list()
        sample_vx=list()
        for x in range(100000):                
            sample_x.append(np.random.uniform(-150,0,1).item())
            sample_vx.append(np.random.uniform(-10,10,1).item())

        sample_x=np.array(sample_x)
        sample_vx=np.array(sample_vx)
        
        self.x_mean=np.mean(sample_x)
        self.vx_mean=np.mean(sample_vx)
        
        self.x_max=np.max(sample_x)
        self.vx_max=np.max(sample_vx)

        self.x_min=np.min(sample_x)
        self.vx_min=np.min(sample_vx)
        
        
    def featurize_state(self,state):
        """
        Returns the featurized representation for a state.
        """
        try:
            state=(state[0].item(),state[1].item())
        except:
            pass
            
        scaled_x = np.divide(np.add(state[0],-self.x_mean),np.add(self.x_max,-self.x_min))
        scaled_vx = np.divide(np.add(state[1],-self.vx_mean),np.add(self.vx_max,-self.vx_min))
        return [scaled_x,scaled_vx]


Agent = RandomAgent