from time import sleep
import numpy as np
from IPython.display import clear_output
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from torch import nn

#Import MiniPacMan environment class definition
from MiniPacManGymv2 import MiniPacManEnv

#Register MiniPacMan in your gymnasium environments
register(
    id="MiniPacMan-v2",
    entry_point=MiniPacManEnv,  # Update with your actual module path
    max_episode_steps=20          # You can also set a default here
)

#Create a MiniPacMan gymnasium environment
env = gym.make("MiniPacMan-v2", render_mode="human", frozen_ghost=False)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6 * 6, 128)  # Flattened board (6x6 → 36)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # Output 4 possible actions (Up, Down, Left, Right)
        self.activation = nn.ReLU()

    def forward(self, state):
        state = state.view(state.shape[0], -1)  # Flatten from (batch_size, 6, 6) → (batch_size, 36)
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation here; raw Q-values
        return x  # Output shape: (batch_size, 4)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return torch.stack(states), actions, torch.tensor(rewards), torch.stack(next_states), torch.tensor(dones)

#set hyperparams -- play with any of these!
gamma=0.9
buffer_size=2000
batch_size=32
num_episodes=5000

Q_target = QNetwork() # initialize target Q network
Q_policy = QNetwork() # initialize policy Q newtwork
Q_optimizer = torch.optim.Adam(Q_policy.parameters(), lr=0.0005)

Q_target.load_state_dict(Q_policy.state_dict()) #copy weights from policy to target

RB=ReplayBuffer(buffer_size) #initialize Replay Buffer
epsilon=1 #initialize epsilon

for e in range(num_episodes):
    new_obs,info=env.reset()
    new_obs=torch.tensor(new_obs,dtype=torch.float32)

    done=False
    truncated=False
    steps=0

    while not done and not truncated: #Loop for one episode
        obs=new_obs

        #choose action
        t=np.random.random()
        if t>epsilon:
            action = torch.argmax(Q_policy(new_obs.unsqueeze(0))).item()
        else:
            action=torch.randint(4,(1,)).item()

        #take a step:
        new_obs,reward, done, truncated, info=env.step(action)
        new_obs=torch.tensor(new_obs,dtype=torch.float32)
        RB.push(obs,action,reward,new_obs,done)
        steps+=1

        if len(RB.buffer)>=batch_size:
            states, actions, rewards, next_states, dones=RB.sample(batch_size)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = rewards.clone().detach() if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, dtype=torch.float32)
            dones = dones.clone().detach() if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.float32)
            preds = Q_policy(states).gather(1, actions.unsqueeze(1)).squeeze()
            targets = rewards + (1 - dones.float()) * gamma * Q_target(next_states).max(1)[0].detach()
            loss = ((targets - preds) ** 2).mean()
            Q_optimizer.zero_grad()
            loss.backward()
            Q_optimizer.step()

    #if reward==20:
        #print("got the ghost!")
    if e%100==0:
        Q_target.load_state_dict(Q_policy.state_dict())

    #reduce episilon if its not too low
    epsilon = max(0.01, epsilon * 0.999)

    #periodic reporting:
    if e>0 and e%100==0:
        print(f'episode: {e}, steps: {steps}, epislon: {epsilon},win: {reward==20}')

print(env.observation_space)
new_obs, info = env.reset()
print(f"Reset observation: {new_obs}, Expected space: {env.observation_space}")
new_obs, reward, done, truncated, info = env.step(action)
print(f"Step observation: {new_obs}, Expected space: {env.observation_space}")

obs, info = env.reset()
done = False
truncated = False

while not done and not truncated:
    env.render()
    obs=torch.tensor(obs,dtype=torch.float32)
    action=torch.argmax(Q_policy(obs.unsqueeze(0))).item()
    obs, reward, done, truncated, info = env.step(action)
    sleep(1)
    clear_output(wait=True)

env.render()
env.close()
