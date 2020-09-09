import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance
import uuid
# example in https://agentmodels.org/chapters/3a-mdp.html
# change state and timeLeft parameters to observe agent's different behavior
___ = ' ' 
D = { 'name' : 'Donut',
      'x' : torch.tensor(2),
      'y' : torch.tensor(5),
      'utility' : torch.tensor(1)
}
S = { 'name' : 'Sushi',
      'x' : torch.tensor(0),
      'y' : torch.tensor(0),
      'utility' : torch.tensor(1)
}
V = { 'name' : 'Veg',
      'x' : torch.tensor(4),
      'y' : torch.tensor(7),
      'utility' : torch.tensor(3)
}
N = { 'name' : 'Noodle',
      'x' : torch.tensor(5),
      'y' : torch.tensor(2),
      'utility' : torch.tensor(2)
}

grid = [
    ['#', '#', '#', '#',  V , '#'],
    ['#', '#', '#', ___, ___, ___],
    ['#', '#',  D , ___, '#', ___],
    ['#', '#', '#', ___, '#', ___],
    ['#', '#', '#', ___, ___, ___],
    ['#', '#', '#', ___, '#',  N ],
    [___, ___, ___, ___, '#', '#'],
    [ S , '#', '#', ___, '#', '#']
]

'''
return True if loc is a restaurant
'''
def got_dst(loc):
    x = loc[0]
    y = len(grid) - loc[1] - 1
    return grid[y][x] != '#' and grid[y][x] != ___

'''
draw the grid, loc: the current location of the agent
'A' - the corrent location of the agent
'#' - the wall
' ' - the path
loc: torch.tensor(x, y)
'''
def print_grid(loc):
    for i, row in enumerate(grid):
        y = len(grid) - i - 1
        print(y, end='|')
        for x, cell in enumerate(row):
            if torch.allclose(torch.tensor((x, y)), loc): # agent location
                print('A', end='')
            elif cell == '#' or cell == ___: # wall
                print(cell, end='')
            else: # restaurant
                print(cell['name'][0], end='')
        print()
    print('  ', end='')
    for i in range(len(grid[0])):
        print(i, end='')
    print()

class Agent():
    '''
    state is the location of the agent
    state : tensor((x, y))
    '''
    def __init__(self, alpha=1000):
        self.alpha = alpha
        self.world_height = len(grid)
        self.world_width = len(grid[0])
        self.all_actions = { 
                                1 : torch.tensor((1, 0)), # move right by 1
                                2 : torch.tensor((-1, 0)), # move left by 1
                                3 : torch.tensor((0, 1)), # move up by 1
                                4 : torch.tensor((0, -1)) # move down by 1
                            }
    
    '''
    return all the possible actions of an agent in a state
    '''
    def get_actions(self, state):
        all_actions = self.all_actions.copy()
        agent_x = state[0]
        agent_y = self.world_height - 1 - state[1]

        if (agent_x == 0) or (grid[agent_y][agent_x - 1] == '#'): # can't move left
            del all_actions[2]
        if (agent_x == self.world_width - 1) or (grid[agent_y][agent_x + 1] == '#') : # can't move right
            del all_actions[1]
        if (agent_y == 0) or (grid[agent_y - 1][agent_x] == '#'): # can't move up
            del all_actions[3]
        if (agent_y == self.world_height - 1) or (grid[agent_y + 1][agent_x] == '#'): # can't move down
            del all_actions[4]

        assert all_actions, "actions at state {} should not be empty".format(state)
        
        return torch.tensor(list(all_actions.keys()))
    '''
    T(state, action) -> state
    assume action is valid
    '''
    def transition(self, state, action):
        return state + self.all_actions[int(action)]
        
    '''
    return the utility at a state
    '''
    def utility(self, state):
        x = state[0]
        y = state[1]
        obj = grid[self.world_height - 1 - y][x]
        assert obj != '#'
        if obj == ___:
            return torch.tensor(-0.1)
        else:
            return obj['utility']
    
    '''
    recursive model
    return the trajectory, a sequence of states
    '''
    def model(self, state, timeLeft, cu_utility=0):
        if (not timeLeft) or (cu_utility > 0): # cu_utility > 0 implies arriving destination
            pyro.factor("state_{}_{}".format(state, str(uuid.uuid1())), self.alpha * cu_utility)
            if timeLeft:
                ignore = torch.tensor([-1, -1]) # invalid state tensor, using for keeping dimension identical 
                return ignore.repeat(timeLeft, 1)
            else: # no time left
                return torch.Tensor()
        # sample an action
        possibleActions = self.get_actions(state)
        num_choices = len(possibleActions)
        action_index = dist.Categorical(torch.tensor([1 / num_choices for _ in range(num_choices)])).sample()
        action = possibleActions[action_index]

        next_state = self.transition(state, action)
        utility = self.utility(state)
        cu_utility += utility
        return torch.cat((state.unsqueeze(0), self.model(next_state, timeLeft - 1, cu_utility)))

    '''
    do inference on the model
    Return the possible trajectory and corresponding probability
    '''
    def infer(self, init_state, timeLeft):
        num_samples = 1000
        action_posterior = Importance(self.model, num_samples = num_samples).run(init_state, timeLeft)
        possib_vals, probs = self.infer_prob(posterior=action_posterior, num_samples=num_samples)
        return possib_vals, probs

    '''
    Sample from a posterior and calculate probabilities of each possible values.
    Return the possible trajectory and corresponding probability
    '''
    def infer_prob(self, posterior, num_samples):
        marginal = EmpiricalMarginal(posterior)
        samples = marginal.sample((num_samples, 1))
        possible_vals, counts = torch.unique(input=torch.flatten(samples, end_dim=1), sorted=True, return_counts=True, dim=0)
        probs = torch.true_divide(counts, num_samples)
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0))
        return possible_vals, probs

    '''
    run the simulation
    '''
    def run(self, init_state, timeLeft):
        possib_traj, probs = self.infer(init_state, timeLeft)
        print("Possible trajectories:", possib_traj)
        print("Probabilities:", probs)
        # sample a trajectory
        traj_index = dist.Categorical(probs=probs).sample()
        traj = possib_traj[traj_index]
        print("Sample trajectory:", traj)
        for state in traj:
            if not torch.all(torch.eq(torch.tensor([-1,-1]).float(), state)): # state is valid
                print_grid(state.long())

if __name__ == "__main__":
    agent = Agent()
    agent.run(init_state=torch.tensor((3,1)),
              timeLeft=10)