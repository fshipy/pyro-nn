import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance

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


'''
Expectation of utilites.
Input:  - utility values,
        - prob of each obtaining utility
Output: - weighted average (expectation)
'''
def expectation(vals, probs):
    ave = np.array(np.average(vals.detach().cpu().numpy(), weights=probs.detach().cpu().numpy()))
    ave = torch.from_numpy(ave).type(dtype=torch.float)
    return ave

class Agent():
    '''
    state is the location of the agent
    state : tensor((x, y))
    '''
    def __init__(self, alpha=100):
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
    
    def expected_utility(self, state, action, timeLeft):
        imd_utility = self.utility(state)
        if timeLeft == 0 or imd_utility > 0:
            return imd_utility
        # infer the posterior utility and the probability distribution
        possib_vals, probs = self.infer_utility(state, action, timeLeft)
        return imd_utility + expectation(possib_vals, probs)

    '''
    Sample from a posterior and calculate probabilities of each possible values.
    '''
    def infer_prob(self, posterior, num_samples, possible_vals=None):
        marginal_dist = EmpiricalMarginal(posterior).sample((num_samples, 1)).float()
        # count the sample for each possible values
        if possible_vals is not None: # infer action probs
            counts = torch.zeros(torch.Size([len(possible_vals)])).float()
            for i in range(len(possible_vals)):
                counts[i] = (marginal_dist == possible_vals[i]).sum()

        else: # infer utility probs
            possible_vals, counts = torch.unique(input=torch.flatten(marginal_dist), sorted=True, return_counts=True)

        probs = torch.true_divide(counts, num_samples)
        # the sum of the probs should be close to 1
        assert torch.allclose(torch.sum(probs), torch.tensor(1.0))
        return possible_vals, probs
    '''
    draw an action and return the action
    add factor by expected utility
    '''
    def action_model(self, state, timeLeft):
        # draw a random action
        possibleActions = self.get_actions(state)
        num_choices = len(possibleActions)
        action_index = dist.Categorical(torch.tensor([1 / num_choices for _ in range(num_choices)])).sample()
        action = possibleActions[action_index]
        # calculate expected uttility for the action
        expected_u = self.expected_utility(state, action, timeLeft)
        # add factor to the action
        pyro.factor("state_{}action_{}".format(state, action), self.alpha * expected_u)
        return action
    
    '''
    draw an action and return the expected utility
    '''
    def utility_model(self, state, action, timeLeft):
        next_state = self.transition(state, action)
        timeLeft = timeLeft - 1
        actions, action_probs = self.infer_actions(next_state, timeLeft)
        next_action_idx = pyro.sample(
            'next_action_state{}_timeleft{}'.format(state, timeLeft),
            dist.Categorical(action_probs)
        )
        next_action = actions[next_action_idx]
        exp_u = self.expected_utility(next_state, next_action, timeLeft)
        return exp_u

    '''
    return the possible actions and possibility to choose each action
    '''
    def infer_actions(self, state, timeLeft):

        action_param_name = 'actions_values_at_state_{}_time{}'.format(state, timeLeft)
        
        # already computed:
        if action_param_name in list(pyro.get_param_store().keys()):
            action_param = pyro.get_param_store().get_param(action_param_name)
            possib_vals = action_param[0]
            probs = action_param[1]
        else:
            num_samples = 10
            action_posterior = Importance(self.action_model, num_samples = num_samples).run(state, timeLeft)
            possib_vals, probs = self.infer_prob(posterior=action_posterior, num_samples=num_samples, possible_vals=self.get_actions(state))
            # cache
            action_param = pyro.param(action_param_name,
                                      torch.cat((possib_vals.unsqueeze(0), 
                                                 probs.unsqueeze(0)),
                                                dim=0))
        return possib_vals, probs
    
    ''' 
    return the possible utilities and possibility to get each utility
    '''
    def infer_utility(self, state, action, timeLeft):

        util_param_name = 'util_state{}_action{}_time{}'.format(state, action, timeLeft) 
        if util_param_name in list(pyro.get_param_store().keys()): 
            util_param = pyro.get_param_store().get_param(util_param_name)
            possib_vals = util_param[0]
            probs = util_param[1]
        else:
            num_samples = 10
            utility_posterior = Importance(self.utility_model, num_samples = num_samples).run(state, action, timeLeft)
            possib_vals, probs = self.infer_prob(utility_posterior, num_samples)
            # cache
            util_param = pyro.param(util_param_name,
                                    torch.cat((possib_vals.unsqueeze(0), 
                                               probs.unsqueeze(0)),
                                               dim=0))
        return possib_vals, probs
    
    '''
    run the simulation
    '''
    def run(self, state, timeLeft):
        print_grid(state)
        if timeLeft:
            possib_actions , probs = self.infer_actions(state, timeLeft)
            best_action_index = dist.Categorical(probs=probs).sample()
            action = possib_actions[best_action_index]
            print("action at state {} and time {} : {}".format(state, timeLeft, self.all_actions[int(action)]))
            state = self.transition(state, action)
            if got_dst(state):
                timeLeft = 1 # will exit in the next recursion
            self.run(state, timeLeft - 1)

if __name__ == "__main__":
    agent = Agent()
    agent.run(state=torch.tensor((3,1)), 
              timeLeft=10)