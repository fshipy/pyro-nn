import numpy as np
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random
import pyro
import pyro.distributions as dist

def is_subset(A, B):
    # return true if A is a subset of B
    A = set(A)
    B = set(B)
    return A.issubset(B)

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class FullMade(nn.Module):
    def __init__(self, input_dim_dict, hidden_sizes, dependency_dict, var_dim_dict, num_masks=1):
        super().__init__()
        
        '''
        input_dim_dict:
        {
            "h" : 8,
            "r" : 1
        }

        var_dim_dict:
        {
            "z1" : 1,
            "y1" : 1,
            "y2" : 1,
            "x1" : 1,
            "k" : 1
        }
        '''
        # TODO, specify and handle random variables type

        self.input_dim_dict = input_dim_dict
        self.var_dim_dict = var_dim_dict
        self.all_var_dim_dict = var_dim_dict.copy()
        self.all_var_dim_dict.update(input_dim_dict)
        self.out_levels = self._construct_output_levels(dependency_dict)
        self.input_levels = self._construct_input_levels(dependency_dict)
        self.hidden_sizes = hidden_sizes
        self.levels = []
        self.input_dims = []
        self.output_dims = []
        for input_level, out_level in zip(self.input_levels, self.out_levels):
            level_nn, input_dim, output_dim = self._build_level(input_level,out_level, dependency_dict)
            self.levels.append(level_nn)
            self.input_dims.append(input_dim)
            self.output_dims.append(output_dim)
        self.levels = nn.ModuleList(self.levels)
        self.softplus = torch.nn.Softplus()
        print("number of levels:", len(self.levels))
        print("input_levels", self.input_levels)
        print("out_levels", self.out_levels)
    
    def forward(self, inDict, i, inVals=None, inNames=None):
        '''
        inVal: [ tensor with shape [n, 8],  tensor with shape [n,1] ]
        inName: ["h", "r"]
        {
            "h" : xxx
            "r" : x
        }
        '''
        ret_var_names_Dict = {}

        #x_in = torch.cat(inVals, dim=-1)
        for input_level, input_dim, out_level, output_dim, level_nn in zip(self.input_levels, self.input_dims, self.out_levels, self.output_dims, self.levels):
            # construct input
            x_in = []
            for in_var in input_level:
                if in_var in inDict:
                    x_in.append(inDict[in_var])
                else:
                    x_in.append(ret_var_names_Dict[in_var])
            x_in = torch.cat(x_in, dim=-1)

            # get output from a level
            x_out = level_nn(x_in)

            start = 0
            # get the distribution parameters for random variables
            # sample them and used as the next inputs
            # TODO we assume all output variables are normal distributed i.e. mean + standard deviation
            for out_var in out_level:
                o = []
                out_var_d = self.all_var_dim_dict[out_var]
                # out_idx = torch.arange(start, end)  # TODO: USE-GPU
                o.append(torch.narrow(x_out, -1, start, out_var_d))
                # get the standard deviation for random variable # TODO fix the indexing (/2 is bad assumption)
                
                assert output_dim % 2 == 0, f"{out_level}, {output_dim}"
                o.append(self.softplus(torch.narrow(x_out, -1, int(start + output_dim / 2), out_var_d)))
                
                sampled_var = self._sample_var(out_var, o, i)
                start += out_var_d
                ret_var_names_Dict[out_var] = sampled_var
        
        return ret_var_names_Dict

    def _sample_var(self, name, o, i):
        # TODO support more distribution types
        # TODO handle repeated random variable name
        return pyro.sample(name+str(i), dist.Normal(o[0], o[1]))

    def _construct_output_levels(self, dependency_dict):
        '''
        given the dependency dictionary, construct the output for each level
        example input:
        {
            "z1" : {"h", "r"},
            "y1" : {"z1", "h"},
            "y2" : {"y1", "z1", "h"},
            "x1" : {"y1", "z1"},
            "k" : {"y1", "z1"}
        }
        output:
        [
            ["z1"],
            ["y1"],
            ["y2", "x1", "k"]
        ]
        '''
        output_levels = []
        unassigned = set(dependency_dict.keys())        
        temp_unassigned = set()
        assigned = set()
        temp_assigned = set()
        while unassigned or temp_unassigned:
            cur_level = []
            while unassigned:
                rand_var = unassigned.pop()
                assign = True
                for dep in dependency_dict[rand_var]:
                    if not (dep in self.input_dim_dict or dep in assigned):
                        assign = False
                        break
                if assign:
                    temp_assigned.add(rand_var)
                    cur_level.append(rand_var)
                else:
                    temp_unassigned.add(rand_var)
            if not cur_level:
                print("Warning: cur_level is empty")
            output_levels.append(cur_level)
            unassigned = temp_unassigned
            assigned = assigned.union(temp_assigned)
            temp_unassigned = set()
            temp_assigned = set()
            # print("cur_level", cur_level)
            # print("unassigned", unassigned)
            # print("assigned", unassigned)
            # print("=====")
        assert len(dependency_dict) == len(assigned)
        return output_levels
    
    def _construct_input_levels(self, dependency_dict):
        '''
        given the dependency dictionary and out levels, construct the input for each level
        
        output:
        [
            ['h', 'r'],
            ['h', 'z1'],
            ['h', 'z1', 'y1']
        ]
        
        '''
        input_levels = [list(self.input_dim_dict.keys())]
        cur_all_inputs = set(self.input_dim_dict.keys())
        for i in range(1, len(self.out_levels)):
            cur_all_inputs = cur_all_inputs.union(set(self.out_levels[i - 1]))
            cur_level_inputs = set()
            for rand_var in cur_all_inputs:
                unused = True
                for rand_var_out in self.out_levels[i]:
                    if rand_var in dependency_dict[rand_var_out]:
                        unused = False
                        break
                if not unused:
                    cur_level_inputs.add(rand_var)
            input_levels.append(list(cur_level_inputs))    
        return input_levels
    
    def _construct_level_ordering(self, input_level, out_level, dependency_dict):
        '''
           construct the sets and orderings for input and output
        '''
        indices = range(len(input_level))
        all_sets = []
        for n in range(1, len(indices) + 1):
            all_sets += list(itertools.combinations(indices, n))
        input_ordering = list(indices) #[(index,) for index in indices]
        # construct out ordering
        out_orderings = []
        for out_rand_var in out_level:
            ord = []
            for dep in dependency_dict[out_rand_var]:
                ord.append(input_level.index(dep))
            ord.sort()
            temp = tuple(ord)
            out_orderings.append(all_sets.index(temp))
        return all_sets, input_ordering, out_orderings
    
    def _build_level(self, input_level, out_level, dependency_dict, hid_sizes=[512], hid_mul=4, hid_layers=1, output_type="normal"):
        '''
           construct the layers and masks based on orderings
        '''
        # TODO output on type, normal, bernoulli...
        all_sets, input_ordering, out_ordering = self._construct_level_ordering(input_level, out_level, dependency_dict)
        hid_sizes = [(len(all_sets) + 1) * hid_mul] *  hid_layers
        def is_subset(A, B):
            # return true if A:int is a subset of B:int
            # A, B are the index of set in all_sets
            A = set(all_sets[A])
            B = set(all_sets[B])
            return A.issubset(B)
        
        is_subset_np = np.frompyfunc(is_subset, 2, 1)
        hid_orderings = [ random.choices(list(range(len(all_sets))), k=hid_size) for hid_size in hid_sizes ]
        level_nn = []
        assert len(hid_sizes) > 0
        expanded_input_ordering = []
        for in_order, in_rand_var in zip(input_ordering, input_level):
            d = self.all_var_dim_dict[in_rand_var]
            expanded_input_ordering += [in_order] * d
        # input layer
        level_nn.append(MaskedLinear(len(expanded_input_ordering), hid_sizes[0]))
        level_nn.append(nn.ReLU())
        masks = [is_subset_np(np.array(expanded_input_ordering)[:, None], np.array(hid_orderings[0])[None, :])] # mask matrix
        expanded_output_ordering = []
        for out_order, out_rand_var in zip(out_ordering, out_level):
            d = self.all_var_dim_dict[out_rand_var]
            expanded_output_ordering += [out_order] * d
        # hidden layers
        for i in range(1, len(hid_sizes)):
            level_nn.append(MaskedLinear(hid_sizes[i-1], hid_sizes[i]))
            level_nn.append(nn.ReLU())
            masks.append(is_subset_np(np.array(hid_orderings[i-1])[:, None], np.array(hid_orderings[i])[None, :]))
        # output layer
        if output_type == "normal":
            expanded_output_ordering *= 2
        level_nn.append(MaskedLinear(hid_sizes[-1], len(expanded_output_ordering)))
        masks.append(is_subset_np(np.array(hid_orderings[-1])[:, None], np.array(expanded_output_ordering)[None, :]))
            
        # set the masks in all MaskedLinear layers
        mask_id = 0
        for i in range(len(level_nn)):
            if isinstance(level_nn[i], MaskedLinear):
                level_nn[i].set_mask(masks[mask_id])
                mask_id += 1
        level_nn = nn.Sequential(*level_nn)
        print("hid_orderings", hid_orderings[0], "size", len(hid_orderings[0]))
        print("expanded_input_ordering", expanded_input_ordering,"size", len(expanded_input_ordering))
        print("expanded_output_ordering", expanded_output_ordering, "size", len(expanded_output_ordering))
        return level_nn, len(expanded_input_ordering), len(expanded_output_ordering)


### ======================================================================================

if __name__ == '__main__':
    # smoke test
    input_dim_dict = {
            "h" : 2,
            "r" : 1
        }
    hidden_sizes = [1, 2, 3]

    var_dim_dict = {
            "z1" : 1,
            "y1" : 1,
            "y2" : 1,
            "x1" : 1,
            "k" : 1
        }
    dependency_dict = {
            "z1" : set(["h", "r"]),
            "y1" : set(["z1", "h"]),
            "y2" : set(["y1", "z1", "h"]),
            "x1" : set(["y1", "h"]),
            "k" : set(["y1", "z1"])
        }
    made = FullMade(input_dim_dict, hidden_sizes, dependency_dict, var_dim_dict)
    print("out_levels", made.out_levels)
    print("input_levels", made.input_levels)
    print(made._construct_level_ordering(made.input_levels[0], made.out_levels[0], dependency_dict))
    print(made._construct_level_ordering(made.input_levels[1], made.out_levels[1], dependency_dict))
    print(made._construct_level_ordering(made.input_levels[2], made.out_levels[2], dependency_dict))
    inp = {
        "h"  : torch.tensor([1., 2.]),
        "r" : torch.tensor([0.5])
    }
    print(made(None, None, inp))