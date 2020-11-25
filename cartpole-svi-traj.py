import math
import gym
import random
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.optim
import pyro.infer
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO

env = gym.make('CartPole-v1')
env.seed(1)
episode = 0
alpha = 200
total_timestamp = 1000

# reset env 
def reset_env():
    global env
    observation = env.reset()
    return observation

# convert raw rewards: [1,1,1,...,1,-10] to 
# expected reward lists
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
    return rewards


# the network to represent policy function p(a|s)
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
        self.policy = Policy() # q(a|s)
        self.target_policy = Policy() # p(a|s)
    
    def guide(self, time_step, actions, states, reward, rewards, epoch=False):
        pyro.module("agentmodel", self)
        for t, state in enumerate(states):
            prob_action = self.policy(state)
            action = pyro.sample("action_%d" %t, dist.Bernoulli(prob_action))
    
    def model(self, time_step, actions, states, reward, rewards, epoch=False):
        '''
        this model takes data of a whole trajectory from simulation and replay
        '''
        pyro.module("agentmodel", self)
        global episode
        episode += 1
        for i in range(len(states)):
            prob = self.target_policy(states[i])
            action = pyro.sample("action_%d" %i, dist.Bernoulli(prob))
            a = 1 if int(actions[i]) == int(action) else -1
            pyro.factor("Episode_{}_{}".format(episode, i), rewards[i] * alpha * a)
        pyro.factor("Episode_{}".format(episode), reward * alpha) # factor by the total reward of the trajectory
        

    # simulate a trajectory and return data    
    def simulate(self):

        observation = reset_env()
        states = []
        actions = []
        total_reward = torch.tensor(0.)
        max_timestamp = 1000
        rewards = []
        for t in range(max_timestamp):
            state = torch.from_numpy(observation).float()
            observation = torch.from_numpy(observation).float()
            action = self.make_sample_action(observation)
            action = round(action.item())
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                reward = -10
            
            rewards.append(torch.tensor(reward).float())
            states.append(state)
            actions.append(action)

            if done:
                print("exit at", t)
                break

        if total_reward < 20:
            total_reward = torch.tensor(0.) # eliminate some “bad” simulations
        
        rewards = generate_rewards(rewards, 0.99) # convert raw rewards to expected rewards

        return [states, actions, total_reward, rewards]
    
    def make_sample_action(self, observation, increment_step=True):
        global steps_done
        sample = random.random()
        # applying epsilon decay when choosing actions
        eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                        math.exp(-1. * steps_done / epsilon_decay)

        if sample > eps_threshold:
            prob = self.policy(observation.float())
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
learning_rate = 2e-5 #1e-5
optimizer = optim.Adam({"lr":learning_rate})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

def optimize(memory):
    num_steps = 2000
    loss = 0
    for t in range(num_steps):
        # make inference on each trajectory's observed data in each step
        for experience in memory:
            states = experience[0]
            actions = experience[1]
            e_reward = experience[2]
            rewards = experience[3]
            loss += svi.step(len(actions), actions, states, e_reward, rewards)
#             if (t % 100 == 0) and (t > 0):
#                 print("at {} step loss is {}".format(t, loss / t))

def train(epoch=2, batch_size=24):
    memory = []
    for epoch in range(epoch):
        print("epoch", epoch)
        pyro.get_param_store().clear() # clear the params to start a new inference
        for i in range(batch_size):
            memory.append(agent.simulate())
        optimize(memory)
        test_loop()
        if (epoch > 0 and epoch % 2 == 0):
            agent.target_policy.load_state_dict(agent.policy.state_dict())
            save()

# test n times and calculate the average result
def test_loop(n=10):
    results = []
    for _ in range(n):
        results.append(test())
    print("Testing %d times, average is %d" %(n, np.array(results).mean()))

# simulate and return how long the agent survives,
# using the agent's policy (q(a|s)) only
def test(max_timestamp=2000, render=False):
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

# save the params
def save():
    print("save to cartpole_model.pt and cartpole_model_params.pt")
    optimizer.save("cartpole_optimzer.pt")
    #torch.save({"model" : policy.state_dict(), "guide" : guide}, "cartpole_model.pt")
    torch.save({"model" : None, "policy" : agent.policy, "steps_done" : steps_done}, "cartpole_model.pt")
    pyro.get_param_store().save("cartpole_model_params.pt")

# load the params
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
test(2000, render=True) # visualize the trained model

''' 
inference results:
epoch 7
exit at 164
exit at 25
exit at 11
exit at 150
exit at 41
exit at 121
exit at 59
exit at 93
exit at 88
exit at 43
exit at 136
exit at 60
exit at 129
exit at 156
exit at 94
exit at 61
exit at 141
exit at 103
exit at 140
exit at 173
exit at 39
exit at 87
exit at 33
exit at 56
testing episode exit at 189
testing episode exit at 104
testing episode exit at 81
testing episode exit at 76
testing episode exit at 138
testing episode exit at 81
testing episode exit at 116
testing episode exit at 143
testing episode exit at 230
testing episode exit at 82
Testing 10 times, average is 124
'''