import math
import gym
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.optim
import pyro.infer
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import random
from pyro.nn import PyroSample
from collections import deque
env = gym.make('CartPole-v1')
env = env.unwrapped
env.seed(1)
episode = 0
alpha = 200
total_timestamp = 1000
init_state = np.array([0.00184833, -0.02882669, 0.01481678, -0.02715952])

# reset env and the initial state 
def reset_env(init_state=init_state):
    global env
    env.reset()
    env = env.unwrapped
    env.state = init_state

# model for policy
# action_took, the actual action that the agent took to achieve the reward
def policy(observation, action_took, reward):
    action = pyro.sample("action", dist.Bernoulli(0.5))
    a = 1 if action_took == action else -1
    pyro.factor("reward", reward * alpha * a) # or torch.exp(reward)
    return action

class Guide(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid())

    def forward(self, observation, action_took, reward):
        pyro.module("guide", self)
        prob = self.neural_net(observation)
        action = pyro.sample("action", dist.Bernoulli(prob))
        return action

guide = Guide()
final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0

# make a prediction using guide
# generate 10 samples and pick the most plausible one
def predict(observation):
    preds = []
    for _ in range(10):
        sampled_guide = guide(observation.float(), None, None)
        preds.append(sampled_guide)

    mean = torch.mean(torch.stack(preds), 0)
    return round(float(mean))

def sample_action(observation):
    global steps_done
    sample = random.random()
    # applying epsilon decay when choosing actions
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)

    if sample > eps_threshold:
        return predict(observation)
    else:
        node_activated = random.randint(0,1)
        steps_done += 1
        return node_activated

# convert raw_rewards [1,1,1,1,..,-10] to expected rewards
def generate_rewards(raw_rewards, gamma):
    rewards = []
    T = len(raw_rewards)
    imme_reward = 1
    reward = 0
    for i in range(0, len(raw_rewards)):
        imme_reward = raw_rewards[T-i-1]
        if i > 0:
            reward = imme_reward + rewards[i-1] * gamma
        else:
            reward = imme_reward
        
        rewards.append(reward)

    rewards = list(reversed(rewards))
    #rewards[-1] = 0
    return rewards

def simulate(max_timestamp=500):
    reset_env()
    observation = init_state
    states = []
    rewards = []
    next_states = []
    actions = []
    for t in range(max_timestamp):
        state = observation
        observation = torch.from_numpy(observation)
        action = sample_action(observation)
        observation, reward, done, _ = env.step(action)

        if done:
            next_state = [0,0,0,0]
            reward = -10
        else:
            next_state = observation
        
        states.append(state)
        rewards.append(reward)
        next_states.append(next_state)
        actions.append(action)

        if done:
            print("exit at", t)
            break
    global episode
    episode += 1
    rewards = generate_rewards(raw_rewards=rewards, gamma=0.98)
    return [states, actions, rewards, next_states]

learning_rate = 1e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(policy, guide, optimizer, loss=Trace_ELBO())
def optimize(memory):
    num_steps = 1000
    for experience in memory:
        states = experience[0]
        actions = experience[1]
        rewards = experience[2]
        for t in range(num_steps):
            loss = 0
            for idx, state in enumerate(states):
                state = torch.from_numpy(state).float()
                action = torch.tensor(actions[idx])
                reward = torch.tensor(rewards[idx])
                loss += svi.step(state, action, reward)

def train(batch_size=10, epoch=2):
    for epoch in range(epoch):
        memory = []
        print("epoch", epoch)
        for _ in range(batch_size):
            memory.append(simulate())
        optimize(memory)
        test_loop()

def test_loop(n=10):
    results = []
    for _ in range(n):
        results.append(test())
    print("Testing %d times, average is %d" %(n, np.array(results).mean()))

def test(max_timestamp=2000, render=False):
    reset_env()
    observation = init_state
    for t in range(max_timestamp):
        if (render):
            env.render()
        observation = torch.from_numpy(observation)
        action = predict(observation)
        observation, reward, done, _ = env.step(action)
        if done:
            print("testing episode exit at", t)
            return t
    print("solve by surviving %d timestamps" %max_timestamp)
    return max_timestamp

train(batch_size=10, epoch=50)
test(2000, render=True) # visualize the trained model
exit(1)