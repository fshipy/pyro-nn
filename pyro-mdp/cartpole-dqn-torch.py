import random
import math
import torch
import gym
import pyro
from torch import nn
import torch.nn.Functional as F
from pyro.nn import PyroModule
from pyro.nn import PyroSample
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.linear1 = PyroModule[nn.Linear](input_dim, 16)
        self.linear2 = PyroModule[nn.Linear](16, 32)
        self.linear3 = PyroModule[nn.Linear](32, 32)
        self.linear4 = PyroModule[nn.Linear](32, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

final_epsilon = 0.05
initial_epsilon = 1
epsilon_decay = 5000
global steps_done
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # applying epsilon decay when choosing actions
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.Tensor(state)
            steps_done += 1
            q_calc = model(state)
            node_activated = int(torch.argmax(q_calc))
            return node_activated
    else:
        node_activated = random.randint(0,1)
        steps_done += 1
        return node_activated

input_dim, output_dim = 4, 2
model = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(model.state_dict())
target_net.eval()
tau = 100
discount = 0.99

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

memory = deque(maxlen=65536) # 2^16
BATCH_SIZE = 128

def sample_experiences():
    mini_batch = random.sample(memory, BATCH_SIZE)
    experiences = [[],[],[],[],[]]
    for row in range(BATCH_SIZE):
        for col in range(5):
            experiences[col].append(mini_batch[row][col])
    return experiences

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0
    experiences = sample_experiences()
    state_batch = torch.Tensor(experiences[0])
    action_batch = torch.LongTensor(experiences[1]).unsqueeze(1)
    reward_batch = torch.Tensor(experiences[2])
    next_state_batch = torch.Tensor(experiences[3])
    done_batch = experiences[4]

    pred_q = model(state_batch).gather(1, action_batch)

    next_state_q_vals = torch.zeros(BATCH_SIZE)

    for idx, next_state in enumerate(next_state_batch):
        if done_batch[idx] == True:
            next_state_q_vals[idx] = -1
        else:
            next_state_q_vals[idx] = (target_net(next_state_batch[idx]).max(0)[0]).detach()

    better_pred = (reward_batch + next_state_q_vals).unsqueeze(1)

    loss = torch.nn.MSELoss()(pred_q, better_pred)
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

#save_state = torch.load("models/DQN_target_11.pth")
#model.load_state_dict(save_state['state_dict'])
#optimizer.load_state_dict(save_state['optimizer'])

env = gym.make('CartPole-v1')
for i_episode in range(5000):
    observation = env.reset()
    episode_loss = 0
    if i_episode % tau == 0:
        target_net.load_state_dict(model.state_dict())
    for t in range(1000):
        #env.render()
        state = observation
        action = select_action(observation)
        observation, reward, done, _ = env.step(action)

        if done:
            next_state = [0,0,0,0]
        else:
            next_state = observation

        memory.append((state, action, reward, next_state, done))
        episode_loss = episode_loss + float(optimize_model())
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode, t+1))
            print("Avg Loss: ", episode_loss / (t+1))
            if (i_episode % 100 == 0):
                eps = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
                print(eps)
            if ((i_episode+1) % 5001 == 0):
                save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(save, "models/DQN_target_" + str(i_episode // 5000) + ".pth")
            break
env.close()