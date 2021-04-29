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
from pyro.nn import PyroSample
from collections import deque
env = gym.make('CartPole-v1')
env.seed(1)
episode = 0
alpha = 200
MAXTIME = 500
init_state = None
num_steps = 100000 # use 10 for smoke test
# reset env and the initial state 
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

    def forward(self, observation):
        prob = self.neural_net(observation)
        assert(prob >= 0 and prob <= 1), "prob should fall inside [0, 1], get: {}, with observation: {}".format(prob.item(), observation)
        return prob

final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0
imme_time = 50
class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = Policy()
        self.target_policy = Policy()
        self.imme_timestamp = 0
        self.echo = False
    
    def guide(self, max_time_step):
        pyro.module("agentmodel", self)

        observation = reset_env()
        for t in range(MAXTIME):
            state = observation
            observation = torch.from_numpy(observation).float()
            prob_action = self.policy(observation)
            action = pyro.sample("action_{}".format(t), dist.Bernoulli(prob_action))
            action = round(action.item())
            observation, reward, done, _ = env.step(action)
            
            if done and self.echo:
                print("guide exit at", t)
                return t
                break
            
            if (done):
                break
        if self.echo:
            print("guide solve the problem at t:", max_time_step)
            return max_time_step
    
    def model(self, max_time_step):
        pyro.module("agentmodel", self)

        observation = reset_env()
        add = True
        total_reward = torch.tensor(0.)
        for t in range(MAXTIME):
            prob = 0.5
            action = pyro.sample("action_{}".format(t), dist.Bernoulli(prob))
            action = round(action.item())
            observation, reward, done, _ = env.step(action)

            if done and add:
                add = False
            
            if add:
                total_reward += reward * 10
                
            if done:
                break

        global episode
        episode += 1
        pyro.factor("Episode_{}".format(episode), total_reward * alpha)

    def run_guide(self):
        global imme_time
        self.echo = True
        results = []
        for _ in range(20):
            global init_state
            init_state = reset_init_state()
            survive = guide(500)
            results.append(survive)
        self.echo = False
        if np.mean(results) > imme_time * 0.9 and imme_time < MAXTIME:
            imme_time = imme_time * 2
            print("update training max_time to", imme_time)

agent = AgentModel()
guide = agent.guide
model = agent.model
learning_rate = 2e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(model, guide, optimizer, loss=TraceGraph_ELBO())

def optimize():
    global imme_time
    loss = 0
    print("Optimizing...")
    for t in range(num_steps):
        global init_state
        init_state = reset_init_state()
        loss += svi.step(imme_time)
        if (t % 1000 == 0) and (t > 0):
            print("at {} step loss is {}".format(t, loss / t))

def train(epoch=2, batch_size=10):
    for epoch in range(epoch):
        optimize()
        agent.run_guide()
        test_loop()
        # if (epoch > 0 and epoch % 3 == 0):
        #     save()

def test_loop(n=10):
    results = []
    for _ in range(n):
        results.append(test())
    print("Testing %d times, average is %d" %(n, np.array(results).mean()))

def test(max_timestamp=2000, render=False):
    observation = env.reset()
    for t in range(max_timestamp):
        if (render):
            env.render()
        observation = torch.from_numpy(observation).float()
        action_prob = agent.policy(observation)
        action = dist.Bernoulli(action_prob).sample()
        observation, reward, done, _ = env.step(int(action))
        if done:
            print("testing episode exit at", t)
            return t
    print("solve by surviving %d timestamps" %max_timestamp)
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
# steps_done = 6000
train(epoch=50)
test(2000, render=False) # visualize the trained model