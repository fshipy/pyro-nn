import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, Importance


# example in https://agentmodels.org/chapters/3-agents-as-programs.html

def expectation(vals, probs):
    """
    Expectation of utilites.
    Input:  - utility values,
            - prob of each obtaining utility
    Output: - weighted average (expectation)
    """
    ave = np.array(np.average(vals.detach().cpu().numpy(), weights=probs.detach().cpu().numpy()))
    ave = torch.from_numpy(ave).type(dtype=torch.float)
    return ave

class Agent():

    def __init__(self, alpha, init_state='0'):
        self.alpha = alpha
        self.init_state = init_state
        self.actions = ["italian", "french"]

    # T(state, action) -> state
    def transition(self, state, action):
        nextStates = ["bad", "good", "spectacular"]
        if action == 0:
            prob = torch.tensor((0.2, 0.6, 0.2))
        else: # french
            prob = torch.tensor((0.05, 0.9, 0.05))
        
        return nextStates[dist.Categorical(prob).sample()]
    
    def utility(self, state):
        table = {
            "bad" : torch.tensor(-10),
            "good": torch.tensor(6),
            "spectacular": torch.tensor(8)
        }
        return table[state]
    
    def expected_utility(self, state, action):
        if not state == '0': # not initial state
            return self.utility(state)
        # if initial state, should return the expected utility for the particular action
        possib_vals, probs = self.infer_utility(state, action)
        return expectation(possib_vals, probs)

    # Sample from a posterior and calculate probabilities of each possible values.
    def infer_prob(self, posterior, possible_vals, num_samples):
        counts = torch.zeros(possible_vals.shape).float()
        marginal_dist = EmpiricalMarginal(posterior).sample((num_samples, 1)).float()
        # count the sample for each possible values
        for i in range(len(possible_vals)):
            counts[i] = (marginal_dist == possible_vals[i]).sum()
        probs = counts/num_samples
        return possible_vals, probs

    def action_model(self, state):
        # draw a random action
        action = dist.Categorical(torch.tensor((0.5, 0.5))).sample()
        # calculate expected uttility for the action
        expected_u = self.expected_utility(state, action)
        # add factor to the action
        pyro.factor("state_%saction_%d" %(state, action), self.alpha * expected_u)
        return action
    
    def utility_model(self, state, action):
        next_state = self.transition(state, action)
        exp_u = self.expected_utility(next_state, action)
        return exp_u

    # return the possible actions and possibility to choose each action
    def infer_actions(self, state):
        num_samples = 50
        action_posterior = Importance(self.action_model, num_samples = num_samples).run(state)
        possib_vals, probs = self.infer_prob(action_posterior, torch.tensor((0, 1)), num_samples)
        return possib_vals, probs
    
    # return the possible utilities and possibility to get each utility
    def infer_utility(self, state, action):
        num_samples = 100
        utility_posterior = Importance(self.utility_model, num_samples = num_samples).run(state, action)
        possib_vals, probs = self.infer_prob(utility_posterior, torch.tensor((-10, 6, 8)), num_samples)
        return possib_vals, probs
    
    def run(self):
        _ , probs = self.infer_actions(self.init_state)
        best_action_index = dist.Categorical(probs=probs).sample()
        print("action:", self.actions[best_action_index])

if __name__ == "__main__":
    agent = Agent(alpha=100)
    agent.run()