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
env = gym.make('CartPole-v1')
env.seed(1)
episode = 0
alpha = 200
MAXTIME = 1000

# reset env and the initial state 
def reset_env():
    global env
    observation = env.reset()
    return observation

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

guide = Policy()
final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0

class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = Policy() # q(a|t)
        self.target_policy = Policy() # p(a|t)
        self.imme_timestamp = 0
        self.states = []
        self.actions = []
    
    def guide(self, max_time_step):
        print(self.states)

        pyro.module("agentmodel", self)
        for t in range(len(self.states)):
            observation = self.states[t]
            prob_action = self.policy(observation)
            pyro.sample("action_{}".format(t), dist.Bernoulli(prob_action))

    def model(self, max_time_step):
        pyro.module("agentmodel", self)
        self.states = []
        observation = reset_env()
        total_reward = torch.tensor(0.)
        for t in range(max_time_step):
            observation = torch.from_numpy(observation).float()
            state = observation

            prob = self.target_policy(observation)
            action = pyro.sample("action_{}".format(t), dist.Bernoulli(prob))
            action = round(action.item())
            observation, reward, done, _ = env.step(action)

            if done:
                reward = -10

            total_reward += reward
            self.states.append(state)

            if done:
                break
        
        global episode
        episode += 1

        if total_reward < 20:
            total_reward = 0 # eliminate some “bad” simulations

        pyro.factor("Episode_{}".format(episode), total_reward * alpha)

    def make_sample_action(self, observation, name=False, increment_step=True):
        global steps_done
        sample = random.random()
        # applying epsilon decay when choosing actions
        eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                        math.exp(-1. * steps_done / epsilon_decay)

        if sample > eps_threshold:
            prob = self.target_policy(observation)
            # if (name):
            #     action = dist.Bernoulli(prob).sample()
            # else:
            action = dist.Bernoulli(prob).sample()
            
            return action
        else:
            action = dist.Bernoulli(0.5).sample()
            if (increment_step):
                steps_done += 1
            return action

agent = AgentModel()
guide = agent.guide
model = agent.model
learning_rate = 1e-5 #1e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

def optimize():
    num_steps = 1000
    loss = 0
    print("Optimizing...")
    for t in range(num_steps):
        loss += svi.step(500)
        # if (t % 50 == 0) and (t > 0):
        #     print("at {} step loss is {}".format(t, loss / t))

def train(epoch=2, batch_size=10):
    for epoch in range(epoch):
    
        optimize()
        test_loop()
        if (epoch > 0 and epoch % 3 == 0):
            agent.target_policy.load_state_dict(agent.policy.state_dict())
            save()

def test_loop(n=10):
    results = []
    for _ in range(n):
        results.append(test())
    print("Testing %d times, average is %d" %(n, np.array(results).mean()))

def test(max_timestamp=500, render=False):
    observation = reset_env()
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

train(epoch=50)
test(2000, render=False) # visualize the trained model
