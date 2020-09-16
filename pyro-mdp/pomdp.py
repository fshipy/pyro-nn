import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance
import uuid
# https://agentmodels.org/chapters/3c-pomdp.html
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
      'utility' : torch.tensor(5)
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
return name if loc is a restaurant
else return False
'''
def is_restaurant(loc):
    x = loc[0]
    y = len(grid) - loc[1] - 1
    obj = grid[y][x]
    if obj != '#' and obj != ___:
        return obj['name']
    else:
        return False

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
return a dictionary that mapping restaurant names with latent states (open or closed)
'''
def map_latent_state(latentStates):
    return {
            'Donut' : latentStates[0],
            'Sushi' : latentStates[1],
            'Veg' : latentStates[2],
            'Noodle' : latentStates[3]
    }

'''
latent states (ground true)
torch.tensor(1) means open
torch.tensor(0) means closed
'''
latent = torch.tensor([1, 1, 1, 0])

'''
another latent state use for priorbelief
'''
alternativeLatent = torch.tensor([1, 0, 1, 1])

def priorlatentStateSampler():
    i = dist.Categorical(probs=torch.tensor([0.8, 0.2])).sample()
    return [alternativeLatent , latent][i]

class Agent():
    '''
    state is the location of the agent
    state : tensor((x, y))
    '''
    def __init__(self, belief, alpha=1000):
        self.belief = belief
        self.alpha = alpha
        self.world_height = len(grid)
        self.world_width = len(grid[0])
        self.all_actions = { 
                                1 : torch.tensor((1, 0)), # move right by 1
                                2 : torch.tensor((-1, 0)), # move left by 1
                                3 : torch.tensor((0, 1)), # move up by 1
                                4 : torch.tensor((0, -1)) # move down by 1
                            }
    
    def _get_obj_by_state(self, state):
        x = state[0]
        y = state[1]
        return grid[self.world_height - 1 - y][x]

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
        
        return list(all_actions.keys())

    '''
    T(state, action) -> state
    assume action is valid
    '''
    def transition(self, state, action):
        proposedNewState = state + self.all_actions[int(action)]
        return proposedNewState
    
    '''
    return all possible actions in state and belief
    if agent thinks a restaurant is closed, it won't enter it
    '''
    def belief_state_to_actions(self, belief, state):
        possibleActions = self.get_actions(state)
        belief_map = map_latent_state(belief())
        for possibleActionIndex in possibleActions:
            advanceState = self.transition(state, possibleActionIndex)
            if is_restaurant(advanceState):
                name = is_restaurant(advanceState)
                belief_status = belief_map[name]
                if not belief_status: # if closed, cannot enter
                    possibleActions.remove(possibleActionIndex)
        return torch.tensor(possibleActions)

    '''
    return all observations at a state
    observations are informing the agent whether a restaurant is closed or open
    valid observations can only be received when state is next to a restaurant
    '''
    def observe(self, state):
        observation = {}
        for possibleActionIndex in self.get_actions(state):
            advanceState = self.transition(state, possibleActionIndex)
            if self.restaurant_status(advanceState) == 'o': # open restaurant
                restaurantName = self._get_obj_by_state(advanceState)['name']
                observation[restaurantName] = torch.tensor(1)
            elif self.restaurant_status(advanceState) == 'c': # closed restaurant
                restaurantName = self._get_obj_by_state(advanceState)['name']
                observation[restaurantName] = torch.tensor(0)
        return observation
    
    #def belief_match_observations
    
    '''
    model for belief
    latentSampler is a probability distribution for belief
    observations are returning values from observe(self, state)
    behave like a conditional statement in 
    '''
    def belief_model(self, latentSampler, observations={}):
        latentSample = latentSampler()
        map_latent = map_latent_state(latentSample)
        if observations:
            for ob_name in list(observations.keys()):
                observed_status = observations[ob_name]
                beliefStatus = map_latent[ob_name]
                distribution = dist.Delta(beliefStatus.float())
                pyro.sample(ob_name, distribution, obs=observed_status.float())
        return latentSample

    def update_belief(self, state):
        # can use cache strategy to speed up running time
        observations = self.observe(state) # observations at state
        if observations: 
            num_samples = 20
            is_belief = Importance(self.belief_model, num_samples = num_samples).run(self.belief, observations)
            is_marginal = EmpiricalMarginal(is_belief)
            self.belief = is_marginal
        # if no observation at the current state(location), skip
        #return belief

    '''
    return 'o' if state is open restaurant
    return 'c' if state is a closed restaurant
    return False if state is not a restaurant
    '''
    def restaurant_status(self, state):
        obj = self._get_obj_by_state(state)
        if (obj != '#' and obj != ___): # is restaurant
            name = obj['name']
            if map_latent_state(latent)[name]:
                return 'o'
            return 'c'
        return False

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
    return the trajectory, a sequence of states, cumulative rewards, and final belief
    '''
    def model(self, state, timeLeft):
        if not timeLeft:
            return torch.Tensor(), 0
        utility = self.utility(state)
        if utility > 0: # arrived destination
            ignore = torch.tensor([-1, -1]) # invalid state tensor, using for keeping dimension identical 
            return torch.cat((state.unsqueeze(0), ignore.repeat(timeLeft - 1, 1))), utility
        # sample an action
        possibleActions = self.belief_state_to_actions(self.belief, state)
        num_choices = len(possibleActions)
        action_index = dist.Categorical(torch.tensor([1 / num_choices for _ in range(num_choices)])).sample()
        action = possibleActions[action_index]
        # make manifest state transition (location)
        next_state = self.transition(state, action)
        self.update_belief(next_state)
        post_traj, post_utils = self.model(next_state, timeLeft - 1)
        return torch.cat((state.unsqueeze(0), post_traj)), utility + post_utils

    def wrapped_model(self, init_state, timeLeft):
        trajectory, cu_utility = self.model(init_state, timeLeft)
        pyro.factor("state_{}_{}".format(init_state, str(uuid.uuid1())), self.alpha * cu_utility)
        return trajectory
    
    '''
    do inference on the model
    Return the possible trajectory and corresponding probability
    '''
    def infer(self, init_state, timeLeft):
        num_samples = 1000
        action_posterior = Importance(self.wrapped_model, num_samples = num_samples).run(init_state, timeLeft)
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
        print("Final Belief:", map_latent_state(self.belief()))

if __name__ == "__main__":
    agent = Agent(belief=priorlatentStateSampler)
    agent.run(init_state=torch.tensor((3,1)),
              timeLeft=10)