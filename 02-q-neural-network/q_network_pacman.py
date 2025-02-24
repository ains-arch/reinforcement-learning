from time import sleep
import numpy as np
from IPython.display import clear_output
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from torch import nn
from MiniPacManGym import MiniPacManEnv

#Register MiniPacMan in your gymnasium environments
register(
    id="MiniPacMan-v0",
    entry_point=MiniPacManEnv,
    max_episode_steps=20
)

#Create a MiniPacMan gymnasium environment
env = gym.make("MiniPacMan-v0", render_mode="human", frozen_ghost=True)

#set hyperparams -- feel free to play with these!
gamma=0.999
num_episodes=10000

#initialize epsilon
epsilon=1

# initialize pytorch neural network
# 2 inputs (horizontal and vertical positions of pacman)
# 4 outputs (one for each possible action)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 16)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(16, 8)
        self.activation = nn.ReLU()
        self.linear3 = nn.Linear(8, 4)

    def forward(self, x):
        # print(f"DEBUG: x: {x}")
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
model=QNetwork()
opt=torch.optim.Adam(model.parameters(),lr=0.01)

# print(f"DEBUG: model: {model}")
# for p in model.parameters():
    # print(f"DEBUG: p: {p}")
    # print(f"DEBUG: p.shape: {p.shape}")

for e in range(num_episodes):
    new_obs,info=env.reset()
    # print(f"DEBUG: new_obs: {new_obs}")
    # print(f"DEBUG: info: {info}")

    new_pos=np.argwhere(new_obs==1)[0] #current pacman position
    new_pos=torch.tensor(new_pos,dtype=torch.float32)
    # print(f"DEBUG: new_pos: {new_pos}")

    done=False
    truncated=False
    steps=0

    while not done and not truncated: #Loop for one episode
        obs=new_obs
        pos=new_pos

        #choose action
        t=np.random.random()
        if t>epsilon:
            action=torch.argmax(model(pos))  #exploitation
            # print(f"DEBUG: exploitation action: {action}")
        else:
            action=np.random.randint(0,4) #exploration
            # print(f"DEBUG: exploration action: {action}")

        #take a step:
        new_obs,reward, done, truncated, info=env.step(action)
        # if reward==10:
            # print("yay!")
        steps+=1
        new_pos=np.argwhere(new_obs==1)[0] #next pacman position
        new_pos=torch.tensor(new_pos,dtype=torch.float32)

        #QNetwork update rule:
        # generate a prediction
        pred = model(pos)[action]
        # generate a target value
        target = reward + (1 - done) * gamma * torch.max(model(new_pos))
        # define a loss
        loss = (target - pred)**2
        # adjust model parameters by backpropagating that loss with pytorch optimizer
        opt.zero_grad()
        loss.backward()
        opt.step()

    #reduce epsilon if its not too low
    #Should be close to zero after 50 - 60% of episodes, and then level off
    epsilon=[epsilon-0.00015 if epsilon > 0.01 else 0.01][0]
    # print(f"DEBUG: epsilon: {epsilon}")

    #periodic reporting:
    if e%100==0:
        # steps should approach six, since that's the most optimal choice
        print(f'episode: {e}, steps: {steps}, epsilon: {epsilon}, win: {reward==10}')

#Run this code cell to see your trained agent in action!
obs, info = env.reset()
done = False
truncated = False

while not done and not truncated:
    env.render()
    pos=np.argwhere(obs==1)[0]  #pacman position
    pos=torch.tensor(pos,dtype=torch.float32)
    action=torch.argmax(model(pos)).item()  #exploitation
    obs, reward, done, truncated, info = env.step(action)
    sleep(1)
    clear_output(wait=True)

env.render()
env.close()
