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
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')
env.seed(1)
episode = 0
alpha = 200
MAXTIME = 200
init_state = None
target_position = 0.5 
num_steps = 30000 # use 10 for smoke test
# reset env and the initial state
total_duration = 0
last_average = MAXTIME
def reset_env():
    global env
    global init_state
    observation = env.reset() # set random init state
    if (init_state is not None): # set a fix init state if not None
        env = env.unwrapped
        env.state = init_state
        observation = init_state
    return observation

def reset_init_state():
    global env
    init_state = env.reset()
    return init_state

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 3),
            nn.Softmax())

    def forward(self, observation):
        prob = self.neural_net(observation)
        return prob

global steps_done
steps_done = 0
class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.target_policy = Policy()
        self.initial_t = round(time.time())
        self.echo = False
        self.results = []
        self.timestamps = []
        self.avg_results = []
        self.save = 1 # if 100 then save
    
    def guide(self):
        pyro.module("agentmodel", self)

        observation = reset_env()
        for t in range(MAXTIME):
            state = observation
            observation = torch.from_numpy(observation).float()
            prob_action = self.policy(observation)
            action = pyro.sample("action_{}".format(t), dist.Categorical(prob_action))
            action = round(action.item())
            observation, reward, done, _ = env.step(action)
            
            if done and self.echo: # solve the problem
                print("guide solve the problem at t =", t)
                return t

            if done:
                break
        
        if self.echo:
            print("guide fail to solve, obs is :", observation)
            return MAXTIME
        
        if (self.save % 10 == 0):
                    self.results.append(t)
                    self.avg_results.append(np.mean(self.results[-10:]))
                    self.timestamps.append(round(time.time()) - self.initial_t)
                    self.save = 1
        else:
                    self.save += 1
        
    
    def model(self):
        pyro.module("agentmodel", self)

        observation = reset_env()
        add = True
        total_reward = torch.tensor(MAXTIME).float()
        solve = False
        for t in range(MAXTIME):
            prob = torch.tensor([1/3, 1/3, 1/3])
            action = pyro.sample("action_{}".format(t), dist.Categorical(prob))
            action = round(action.item())
            observation, reward, done, info = env.step(action)
                
            if done:
                if (t < MAXTIME - 1):
                    solve = True
                break

        if solve:
            total_reward -= t
        else:
            total_reward = np.abs(observation[0] - (-target_position))

        global episode
        episode += 1
        
        pyro.factor("Episode_{}".format(episode), total_reward * alpha)


    def run_guide(self):
        self.echo = True
        results = []
        for _ in range(20):
            global init_state
            init_state = reset_init_state()
            survive = guide()
            results.append(survive)
        self.echo = False

agent = AgentModel()
guide = agent.guide
model = agent.model
learning_rate = 2e-3 #1e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

def optimize():
    loss = 0
    print("Optimizing...")
    for t in range(num_steps):
        global init_state
        init_state = reset_init_state()
        loss += svi.step()
        if (t % 100 == 0) and (t > 0):
            print("at {} step loss is {}".format(t, loss / t))

def train(epoch=2, batch_size=10):
    global start_time
    global total_duration
    for epoc in range(epoch):
        pyro.get_param_store().clear()
        optimize()
        agent.run_guide()
        test_loop(n=100)
        print("epoch end", time.time())
        cycle_duration = time.time() - start_time
        print("cycle duration", cycle_duration)
        
        total_duration += cycle_duration
        print("total duration", total_duration)
        
        start_time = time.time()
        plt.plot(agent.timestamps, agent.results)
        plt.plot(agent.timestamps, agent.avg_results)
        plt.show()

def test_loop(n=100):
    results = []
    for _ in range(n):
        results.append(test())
    last_average = np.array(results).mean()
    print("Testing %d times, average is %d" %(n, last_average))

def test(max_timestamp=MAXTIME, render=False):
    observation = env.reset()
    for t in range(max_timestamp):
        if (render):
            env.render()
        observation = torch.from_numpy(observation).float()
        action_prob = agent.policy(observation)
        action = dist.Categorical(action_prob).sample()
        observation, reward, done, _ = env.step(int(action))
        if done:
            print("solve at", t)
            return t
    print("fail to dolve, obs is", observation)
    return max_timestamp

def save():
    print("save to cartpole_model.pt and cartpole_model_params.pt")
    optimizer.save("cartpole_optimzer.pt")
    #torch.save({"model" : policy.state_dict(), "guide" : guide}, "cartpole_model.pt")
    torch.save({"model" : None, "policy" : agent.policy, "steps_done" : steps_done}, "cartpole_model.pt")
    pyro.get_param_store().save("cartpole_model_params.pt")

def load():
    print("load from cartpole_model.pt and cartpole_model_params.pt")
    saved_model_dict = torch.load("cartpole_model.pt")
    #policy.load_state_dict(saved_model_dict['model'])
    agent.policy.load_state_dict(saved_model_dict['policy'].state_dict())
    pyro.get_param_store().load("cartpole_model_params.pt")
    #optimizer.load("cartpole_optimzer.pt")
    #pyro.module('guide', guide, update_module_params=True)
    #steps_done = saved_model_dict['steps_done']

# load()

steps_done = 6000
start_time = time.time()
print("start at", datetime.now())
train(epoch=1)
test(MAXTIME, render=False) # visualize the trained model
env.close()
print("end at", datetime.now())