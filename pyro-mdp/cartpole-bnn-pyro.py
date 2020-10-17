import random
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import gym
import pyro
import pyro.infer
import pyro.optim as optim
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from collections import deque

class DQN(PyroModule):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.linear1 = PyroModule[nn.Linear](input_dim, 16)
        self.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([16, input_dim]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 10.).expand([16]).to_event(1))
        self.linear2 = PyroModule[nn.Linear](16, 32)
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([32, 16]).to_event(2))
        self.linear2.bias = PyroSample(dist.Normal(0., 10.).expand([32]).to_event(1))
        self.linear3 = PyroModule[nn.Linear](32, 32)
        self.linear3.weight = PyroSample(dist.Normal(0., 1.).expand([32, 32]).to_event(2))
        self.linear3.bias = PyroSample(dist.Normal(0., 10.).expand([32]).to_event(1))
        self.linear4 = PyroModule[nn.Linear](32, output_dim)
        self.linear4.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, 32]).to_event(2))
        self.linear4.bias = PyroSample(dist.Normal(0., 10.).expand([output_dim]).to_event(1))

    def forward(self, x, obs=None):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        # x = self.linear4(x)
        out = pyro.sample("actionOut", dist.Normal(x, 1.), obs=obs)
        return out

final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.Tensor(state)
            steps_done += 1
            q_calc = predict(state, guide)
            node_activated = int(torch.argmax(q_calc))
            return node_activated
    else:
        node_activated = random.randint(0,1)
        steps_done += 1
        return node_activated

input_dim, output_dim = 4, 2
model = DQN(input_dim, output_dim)
guide = AutoDiagonalNormal(model)
target_guide = AutoDiagonalNormal(model)
tau = 100
discount = 0.99

learning_rate = 1e-5
optimizer = optim.Adam({"lr":learning_rate})
memory = deque(maxlen=65536)
BATCH_SIZE = 512

def sample_experiences():
    mini_batch = random.sample(memory, BATCH_SIZE)
    experiences = [[],[],[],[],[]]
    for row in range(BATCH_SIZE):
        for col in range(5):
            experiences[col].append(mini_batch[row][col])
    return experiences


def predict(x, guide):
    preds = []
    for _ in range(10):
        guide_trace = poutine.trace(guide).get_trace(x)
        preds.append(poutine.replay(model, guide_trace)(x, None))  
    mean = torch.mean(torch.stack(preds), 0) 
    return mean

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    experiences = sample_experiences()
    state_batch = torch.Tensor(experiences[0])
    action_batch = torch.LongTensor(experiences[1]).unsqueeze(1)
    reward_batch = torch.Tensor(experiences[2])
    next_state_batch = torch.Tensor(experiences[3])
    done_batch = experiences[4]

    next_state_q_vals = torch.zeros(BATCH_SIZE)

    for idx, next_state in enumerate(next_state_batch):
        if done_batch[idx] == True:
            next_state_q_vals[idx] = -1
        else:
            # .max in pytorch returns (values, idx), we only want vals
            next_state_q_vals[idx] = (predict(next_state, guide).max(0)[0]).detach()


    better_preds = (reward_batch + next_state_q_vals).unsqueeze(1)
    
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    num_iterations = 10
    total_loss = 0
    loss = 0
    # global temp_flag
    # temp_flag = None
    print("len:", len(memory))

    for i in range(num_iterations):
        for batch_id, state in enumerate(state_batch):
            better_pred = better_preds[batch_id]
            # calculate the loss and take a gradient step
            loss += svi.step(state, better_pred)
        
        print("in {} epoch, loss:".format(i), loss / BATCH_SIZE)
        total_loss += loss
        loss = 0
    target_guide = guide
    return total_loss / (num_iterations * BATCH_SIZE)  # sum of loss in a batch 

# points = []
# losspoints = []

#temp_flag = 1
time_to_optimize = BATCH_SIZE / 2
env = gym.make('CartPole-v1')
for i_episode in range(5000):
    observation = env.reset()
    episode_loss = 0
    #if i_episode % tau == 0:
        #target_guide = guide
    for t in range(1000):
        #env.render()
        state = observation
        action = select_action(observation)
        observation, reward, done, _ = env.step(action)

        if done:
            next_state = [0,0,0,0]
        else:
            next_state = observation
        # if (temp_flag):
        memory.append((state, action, reward, next_state, done))
        if not time_to_optimize:
            episode_loss = episode_loss + float(optimize_model())
            time_to_optimize = BATCH_SIZE / 2
        else:
            time_to_optimize -= 1
        if done:
            #points.append((i_episode, t+1))
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Avg Loss: ", episode_loss / (t+1))
            #losspoints.append((i_episode, episode_loss / (t+1)))
            if (i_episode % 100 == 0):
                eps = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
                print(eps)
            if ((i_episode+1) % 5001 == 0):
                save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(save, "models/DQN_target_" + str(i_episode // 5000) + ".pth")
            break
env.close()
