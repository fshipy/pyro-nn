import gym
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.optim
import pyro.infer
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance
import uuid
import random

env = gym.make('CartPole-v1')
env = env.unwrapped
env.seed(1)
episode = 0
alpha = 200
total_timestamp = 1000
init_state = np.array([0.00184833, -0.02882669, 0.01481678, -0.02715952])

def model(trajectory_sampler, actionSampler, imm_timestamp):
    total_reward = 0
    #observation = env.reset()
    env.state = init_state
    observation = init_state
    trajectory = torch.from_numpy(np.random.randint(-1, 0, total_timestamp))
    for t in range(imm_timestamp):
        #env.render()
        action = int(pyro.sample("action{}".format(observation), dist.Bernoulli(0.5)))
        if (trajectory_sampler):
            sample_traj = int(trajectory_sampler.sample()[t])
            if sample_traj > -1:
                action = sample_traj
        
        observation, reward, done, info = env.step(action)
    
        total_reward += reward
        trajectory[t] = action
        if done:
            print("exit at", t, end=" ")
            break
    global episode
    if total_reward < imm_timestamp * 0.96:
        total_reward = 0 # eliminate some “bad” simulations
    else:
        total_reward = total_reward / imm_timestamp
    print("reward", total_reward)
    pyro.factor("Episode_{}".format(episode), total_reward * alpha)
    episode += 1
    return trajectory

def infer():
    action_sampler = env.action_space
    trajectory_sampler = None
    imm_timestamp = 30
    num_samples = 50
    for i in range(50):
        posterior = Importance(model, num_samples = num_samples).run(trajectory_sampler, action_sampler, imm_timestamp)
        trajectory_sampler = EmpiricalMarginal(posterior)
        samples = trajectory_sampler.sample((num_samples, 1))
        possible_vals, counts = torch.unique(input=torch.flatten(samples, end_dim=1), sorted=True, return_counts=True, dim=0)
        probs = torch.true_divide(counts, num_samples)
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0))
        print("possible_Traj")
        print(possible_vals)
        print(probs)
        imm_timestamp+=10
        num_samples+=50
        print("in {}th inference, sample traj is".format(i), trajectory_sampler.sample())
    return trajectory_sampler
    
trajectory = infer().sample()
print("last trial:", trajectory)
#observation = env.reset()
env.state = state
observation = state
actionlist1 = []
for t in range(total_timestamp):
    env.render()
    action = int(trajectory[t])
    if action < 0:
        action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    actionlist1.append(action)
    if done:
        print("break at ", t)
        break
env.close()
print(actionlist1)
