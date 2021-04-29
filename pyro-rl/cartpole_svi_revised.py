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
import time
import matplotlib.pyplot as plt
env = gym.make('CartPole-v1')
env.seed(1)
episode = 0
alpha = 200
MAXTIME = 1000
init_state = None
num_steps = 3000 # use 10 for smoke test
# reset env and the initial state
total_duration = 0
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
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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
        self.initial_t = round(time.time())
        self.echo = False
        self.results = []
        self.timestamps = []
        self.avg_results = []
    
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
            
            if done:
                self.results.append(t)
                self.avg_results.append(np.mean(self.results[-10:]))
                self.timestamps.append(round(time.time()) - self.initial_t)
                return t
        
        # solved
        self.results.append(max_time_step)
        self.avg_results.append(np.mean(self.results[-10:]))
        self.timestamps.append(round(time.time()) - self.initial_t)
        
        if self.echo:
            print("guide solve the problem at t:", max_time_step)
            return max_time_step
    
    def model(self, max_time_step):
        #print("in model")
        pyro.module("agentmodel", self)

        observation = reset_env()
        add = True
        # states = []
        # actions = []
        total_reward = torch.tensor(0.)
        for t in range(MAXTIME):
            prob = 0.5
            action = pyro.sample("action_{}".format(t), dist.Bernoulli(prob))
            action = round(action.item())
            #action = self.sample_action(observation, name="action_%d" %t)
            observation, reward, done, _ = env.step(action)

            if done and add:
                add = False
            
            if add:
                total_reward += reward * 10
                
            if done:
                break

        global episode
        episode += 1
        
        if total_reward < max_time_step * 0.5:
            total_reward = 0.01 # eliminate some “bad” simulations

        pyro.factor("Episode_{}".format(episode), total_reward * alpha)

    def run_guide(self):
        global imme_time
        self.echo = True
        results = []
        for _ in range(20):
            global init_state
            init_state = reset_init_state()
            survive = guide(MAXTIME)
            results.append(survive)
        self.echo = False

agent = AgentModel()
guide = agent.guide
model = agent.model
learning_rate = 8e-4 #1e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

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
    # memory = []
    global start_time
    global total_duration
    for epoc in range(epoch):
        pyro.get_param_store().clear()
        
        optimize()
        agent.run_guide()
        test_loop()
        print("epoch end", time.time())
        cycle_duration = time.time() - start_time
        print("cycle duration", cycle_duration)
        
        total_duration += cycle_duration
        print("total duration", total_duration)
        
        start_time = time.time()
        plt.plot(agent.timestamps, agent.results)
        plt.plot(agent.timestamps, agent.avg_results)
        plt.xlabel("timestamp /sec")
        plt.ylabel("score")
        plt.show()
        # if (epoch > 0 and epoch % 3 == 0):
        #     agent.target_policy.load_state_dict(agent.policy.state_dict())
        #     save()

def test_loop(n=10):
    results = []
    for _ in range(n):
        results.append(test())
    print("Testing %d times, average is %d" %(n, np.array(results).mean()))

def test(max_timestamp=MAXTIME, render=False):
    global init_state
    init_state = reset_init_state()
    observation = reset_env()
    for t in range(max_timestamp):
        if (render):
            env.render()
        observation = torch.from_numpy(observation).float()
        action_prob = agent.policy(observation)
        action = dist.Bernoulli(action_prob).sample()
        # action = agent.predict(observation, "policy")
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
start_time = time.time()
print("time start", start_time)
train(epoch=1)
test(2000, render=False) # visualize the trained model