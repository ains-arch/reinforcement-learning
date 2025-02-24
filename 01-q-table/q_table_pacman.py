from time import sleep
import numpy as np
from IPython.display import clear_output
import gymnasium as gym
from gymnasium.envs.registration import register
from MiniPacManGym import MiniPacManEnv

#Register MiniPacMan in your gymnasium environments
register(
    id="MiniPacMan-v0",
    entry_point=MiniPacManEnv,
    max_episode_steps=20
)

#Create a MiniPacMan gymnasium environment
env = gym.make("MiniPacMan-v0", render_mode="human", frozen_ghost=True)

#set hyperparams
gamma=0.999
alpha=0.9
num_episodes=10000

#initialize epsilon, Q
epsilon=1
Q=np.zeros((6,6,4)) #First two coordinates encode state, last encodes action
# print(f"DEBUG: Q[1][1].argmax(axis=0): {Q[1][1].argmax(axis=0)}")

for e in range(num_episodes):
    new_obs,info=env.reset()
    # print(f"DEBUG: new_obs: {new_obs}")
    # print(f"DEBUG: info: {info}")
    new_pos=np.argwhere(new_obs==1)[0] #current pacman position
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
            action=Q[pos[0], pos[1]].argmax(axis=0) #exploitation
            # print(f"DEBUG: exploitation action: {action}")
        else:
            action=np.random.randint(0,4) #exploration
            # print(f"DEBUG: exploration action: {action}")

        #take a step:
        new_obs,reward, done, truncated, info=env.step(action)
        steps+=1
        new_pos=np.argwhere(new_obs==1)[0] #next pacman position

        #Q-table update rule:
        Q[pos[0],pos[1],action]=Q[pos[0],pos[1],action] - alpha * (Q[pos[0],pos[1],action] - [reward + gamma * Q[new_pos[0], new_pos[1]].max()])

    #reduce epsilon if its not too low
    #Should be close to zero after 50 - 60% of episodes, and then level off
    epsilon=[0.999*epsilon if epsilon > 0.01 else 0.01][0]
    # print(f"DEBUG: epsilon: {epsilon}")

#periodic reporting:
if e%100==0:
    print(f'episode: {e}, steps: {steps}, epsilon: {epsilon}, win: {reward==10}')

#Run this code cell to see your trained agent in action!
obs, info = env.reset()
done = False
truncated = False

while not done and not truncated:
    env.render()
    pos=np.argwhere(obs==1)[0]  #pacman position
    action=Q[pos[0], pos[1]].argmax(axis=0)
    obs, reward, done, truncated, info = env.step(action)
    sleep(1)
    clear_output(wait=True)

env.close()
