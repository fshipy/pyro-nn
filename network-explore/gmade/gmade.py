import numpy as np
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import random
import pyro
import pyro.distributions as dist

# def is_subset(A, B):
#     # return true if A is a subset of B
#     A = set(A)
#     B = set(B)
#     return A.issubset(B)

"""
Hyperparameters:

hidden_sizes
hidden_layers
"""


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class GMADE(nn.Module):
    def __init__(
        self,
        input_dim_dict,
        dependency_dict,
        var_dim_dict,
        dist_type_dict={},
        to_event_dict={},
        hidden_sizes=4,
        hidden_layers=1,
        verbose=True,
        use_cuda=False,
    ):
        """
        A generalized MADE variant

        Variables:

        input_dim_dict (dict) : a dictionary <str, int> where the key represents the name of input
                                and value represents the dimension of the input
        dependency_dict (dict) : a dictionary <str, list/set(str)>  where the key represents the name of
                                a random variable and the value represents its dependency
        var_dim_dict (dict) : a dictionary <str, int> where the key represents the name of a random
                                variable and value represents the dimension of the random variable
                                The dimension is the size after pyro.sample statement at axis 0
        dist_type_dict (dict) : a dictionary <str, str/tuple> where the key represents the name of a random
                                variable and the value represents the type of correspnding distribution used in pyro.sample
                                Unspecified random variable will be set to normal by default
                                supported distributions:
                                Normal, Bernoulli, Categorical
                                if the type is Categorical, value should be ('cate', num_class)
                                TODO: we can use enumeration instead of str
        to_event_dict (dict) : a dictionary <str, int> used if tensor has declared dependency 
                                if declared, distribution.to_event(n) will be specified when sampling
        hidden_sizes (int/list(int)/list(list(int))) : there are two ways to set hidden layers:
                                1. specify a multiplier and use the product with the total number of
                                sets in the level as the dimension of the hidden layer (recommended)
                                if hidden_sizes is int, then every level will use the same multiplier
                                if hidden_sizes if list(int), level i will use multiplier hidden_sizes[i] in
                                for each level
                                2. fully customized hidden dimensions, hidden_sizes should be a 2d list in this case
                                Note: if hidden_sizes is list type, the user needs ensure the length of the list equals
                                to the number of levels
        hidden_layers (int/list(int)) : if int, each level will have the same number of hidden layers,
                                if list(int), level i will have hidden_layers[i] number of hidden layers

        Examples:
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
        dependency_dict = {
            "z1" : set(["h", "r"]),
            "y1" : set(["z1", "h"]),
            "y2" : set(["y1", "z1", "h"]),
            "x1" : set(["y1", "h"]),
            "k" : set(["y1", "z1"])
        }
        dist_type_dict = {
            "z1" : "norm",
            "y1" : "norm",
            "y2" : "bern",
            "x1" : ("cate", 10)
            "k" : "norm"
        }
        to_event_dict = {
            "z1" : 1
            # others will be assumed to None
        }
        hidden_sizes = 4
        hidden_layers = 1
        """
        super().__init__()

        self.input_dim_dict = input_dim_dict
        self.var_dim_dict = var_dim_dict
        self.all_var_dim_dict = var_dim_dict.copy()
        self.all_var_dim_dict.update(input_dim_dict)
        self.out_levels = self._construct_output_levels(dependency_dict)
        self.input_levels = self._construct_input_levels(dependency_dict)
        self.hidden_sizes = hidden_sizes
        self.hidden_layers = hidden_layers
        self.verbose = verbose
        self.to_event_dict = to_event_dict
        self.usePyroSample = True
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
        # fill non missed dist type
        for var_k in var_dim_dict.keys():
            if var_k not in dist_type_dict:
                dist_type_dict[var_k] = "norm"

        self.dist_type_dict = dist_type_dict

        self.levels = []
        self.input_dims = []
        self.output_dims = []
        for i, (input_level, out_level) in enumerate(
            zip(self.input_levels, self.out_levels)
        ):
            hid_layers = hidden_layers if isinstance(hidden_layers, int) else hidden_layers[i]
            if isinstance(hidden_sizes, int):
                # all levels have the same hidden_size multiplier, recommended
                level_nn, input_dim, output_dim = self._build_level(
                    input_level, out_level, dependency_dict, hid_mul=hidden_sizes, hid_layers=hid_layers
                )
            elif isinstance(hidden_sizes[i], int):
                # using multiplier, but different for each level
                level_nn, input_dim, output_dim = self._build_level(
                    input_level, out_level, dependency_dict, hid_mul=hidden_sizes[i], hid_layers=hid_layers
                )
            elif isinstance(hidden_sizes[i], list):
                # custom hidden layer
                level_nn, input_dim, output_dim = self._build_level(
                    input_level,
                    out_level,
                    dependency_dict,
                    hid_mul=None,
                    hid_sizes=hidden_sizes[i],
                )
            else:
                raise ValueError(
                    "hidden_sizes are not well defined. hidden_sizes={}".format(
                        hidden_sizes
                    )
                )
            self.levels.append(level_nn)
            self.input_dims.append(input_dim)
            self.output_dims.append(output_dim)
        self.levels = nn.ModuleList(self.levels)
        self.softplus = torch.nn.Softplus()
        
        if verbose:
            print("number of levels:", len(self.levels))
            print("input_levels", self.input_levels)
            print("out_levels", self.out_levels)

    def forward(self, inDict, suffix=None, varNameMap=None):
        """
        Variables:

        inDict (dict) : a dictionary <str, tensor> that specified the input to the model
        suffix (str) : the suffix to add to name of random variables, used in pyro.sample statement
                       if not used in iterative or recursive functions, can set it to "" (default when None)
        varNameMap (dict) : a substitute for suffix, used when the user want to specify the full name for
                            random variable name used in pyro.sample statement (not frequently used)

        Return:
        ret_var_names_Dict (dict) : a dictionary <str, tensor> specified the output for each random variable 
                                    (output after pyro.sample)

        Example:

        inDict = {
            "h" : torch.tensor([1,2,3])
            "r" : torch.tensor(2)
        }
        suffix = "_1"
        """
        if suffix is None and varNameMap is None:
            print(
                "Warning, both suffix and varNameMap are not provided, default suffix is empty string"
            )
            suffix = ""

        ret_var_names_Dict = {}

        # x_in = torch.cat(inVals, dim=-1)
        for input_level, input_dim, out_level, output_dim, level_nn in zip(
            self.input_levels,
            self.input_dims,
            self.out_levels,
            self.output_dims,
            self.levels,
        ):
            # construct input
            x_in = []
            for in_var in input_level:
                if in_var in inDict:
                    in_val = (
                        inDict[in_var]
                        if inDict[in_var].dim() > 0
                        else torch.unsqueeze(inDict[in_var], 0)
                    )
                    x_in.append(in_val)
                else:
                    x_in.append(ret_var_names_Dict[in_var])
            x_in = torch.cat(x_in, dim=-1)#.to(self.device)

            # get output from a level
            x_out = level_nn(x_in)

            start = 0
            # get the distribution parameters for random variables
            # sample them and used as the next inputs
            for out_var in out_level:
                o = []
                out_var_d = self.all_var_dim_dict[out_var]
                out_var_type = "norm"
                toEvent = None if out_var not in self.to_event_dict else self.to_event_dict[out_var]

                if self.dist_type_dict[out_var] == "norm":
                    o.append(torch.narrow(x_out, -1, start, out_var_d))
                    start += out_var_d
                    # get the standard deviation for random variable
                    o.append(self.softplus(torch.narrow(x_out, -1, start, out_var_d)))
                    start += out_var_d
                elif self.dist_type_dict[out_var] == "bern":
                    out_var_type = "bern"
                    o.append(torch.narrow(x_out, -1, start, out_var_d))
                    start += out_var_d
                elif isinstance(self.dist_type_dict[out_var], tuple) and self.dist_type_dict[out_var][0] == "cate":
                    out_var_type = "cate"
                    num_class = self.dist_type_dict[out_var][1]
                    cat_out = torch.narrow(x_out, -1, start, out_var_d * num_class)
                    start += out_var_d * num_class
                    if cat_out.dim() > 1: # has batch
                        cat_out = cat_out.view(-1, out_var_d, num_class)
                    else:
                        cat_out = cat_out.view(out_var_d, num_class)
                    # FIXME do we need to apply softmax / logsoftmax ? 
                    o.append(cat_out)

                if suffix is not None:
                    varName = out_var + str(suffix)
                else:
                    varName = varNameMap[out_var]
                sampled_var = self._sample_var(varName, o, type=out_var_type, toEvent=toEvent)
                ret_var_names_Dict[out_var] = sampled_var
                # TODO input last output layer as next input or sampled value
                # i.e. if normal, should next input be [loc, scale] or value sampled from dist 

        return ret_var_names_Dict
    
    def train(self, mode=True):
        super().train(mode)
        self.usePyroSample = True

    def eval(self):
        super().eval()
        self.usePyroSample = False

    def _sample_var(self, name, o, type="norm", toEvent=None):
        # TODO support more distribution types
        # TODO handle repeated random variable name
        if type == "norm":
            if not self.usePyroSample: 
                print("haha")
                return dist.Normal(o[0], o[1]).sample()
            if toEvent: return pyro.sample(name, dist.Normal(o[0], o[1]).to_event(toEvent))
            else: return pyro.sample(name, dist.Normal(o[0], o[1]))
        elif type == "bern":
            if not self.usePyroSample: return dist.Bernoulli(o[0]).sample()
            if toEvent: return pyro.sample(name, dist.Bernoulli(o[0]).to_event(toEvent))
            else: return pyro.sample(name, dist.Bernoulli(o[0]))
        elif type == "cate":
            if not self.usePyroSample: return dist.Categorical(o[0]).sample()
            if toEvent: return pyro.sample(name, dist.Categorical(o[0]).to_event(toEvent))
            else: return pyro.sample(name, dist.Categorical(o[0]))

    def _construct_output_levels(self, dependency_dict):
        """
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
        """
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
        """
        given the dependency dictionary and out levels, construct the input for each level

        output:
        [
            ['h', 'r'],
            ['h', 'z1'],
            ['h', 'z1', 'y1']
        ]

        """
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
        """
        construct the sets and orderings for input and output of a level
        """
        indices = range(len(input_level))
        all_sets = []
        for n in range(1, len(indices) + 1):
            all_sets += list(itertools.combinations(indices, n))
        input_ordering = list(indices)  # [(index,) for index in indices]
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

    def _build_level(
        self,
        input_level,
        out_level,
        dependency_dict,
        hid_mul=4,
        hid_layers=1,
        hid_sizes="auto",
    ):
        """
        construct the layers and masks based on orderings
        """

        def is_subset(A, B):
            # return true if A:int is a subset of B:int
            # A, B are the index of set in all_sets
            # used for np.frompyfunc
            A = set(all_sets[A])
            B = set(all_sets[B])
            return A.issubset(B)

        all_sets, input_ordering, out_ordering = self._construct_level_ordering(
            input_level, out_level, dependency_dict
        )
        if hid_sizes == "auto":
            hid_sizes = [(len(all_sets) + 1) * hid_mul] * hid_layers

        is_subset_np = np.frompyfunc(is_subset, 2, 1)
        hid_orderings = [
            random.choices(list(range(len(all_sets))), k=hid_size)
            for hid_size in hid_sizes
        ]
        level_nn = []
        assert len(hid_sizes) > 0
        expanded_input_ordering = []
        for in_order, in_rand_var in zip(input_ordering, input_level):
            d = self.all_var_dim_dict[in_rand_var]
            expanded_input_ordering += [in_order] * d
        # input layer
        level_nn.append(MaskedLinear(len(expanded_input_ordering), hid_sizes[0]))
        level_nn.append(nn.ReLU())
        masks = [
            is_subset_np(
                np.array(expanded_input_ordering)[:, None],
                np.array(hid_orderings[0])[None, :],
            )
        ]  # mask matrix
        expanded_output_ordering = []
        for out_order, out_rand_var in zip(out_ordering, out_level):
            if self.dist_type_dict[out_rand_var] == "norm":
                t = 2
            elif self.dist_type_dict[out_rand_var] == "bern":
                t = 1
            elif isinstance(self.dist_type_dict[out_rand_var], tuple) and (
                self.dist_type_dict[out_rand_var][0] == "cate"
            ):
                t = self.dist_type_dict[out_rand_var][1]
            else:
                raise ValueError(
                    "Does not support this type of distrbution yet: {}".format(
                        self.dist_type_dict[out_rand_var]
                    )
                )

            d = self.all_var_dim_dict[out_rand_var]
            expanded_output_ordering += [out_order] * d * t
        # hidden layers
        for i in range(1, len(hid_sizes)):
            level_nn.append(MaskedLinear(hid_sizes[i - 1], hid_sizes[i]))
            level_nn.append(nn.ReLU())
            masks.append(
                is_subset_np(
                    np.array(hid_orderings[i - 1])[:, None],
                    np.array(hid_orderings[i])[None, :],
                )
            )
        # output layer
        # if output_type == "normal":
        #     expanded_output_ordering *= 2
        level_nn.append(MaskedLinear(hid_sizes[-1], len(expanded_output_ordering)))
        masks.append(
            is_subset_np(
                np.array(hid_orderings[-1])[:, None],
                np.array(expanded_output_ordering)[None, :],
            )
        )

        # set the masks in all MaskedLinear layers
        mask_id = 0
        for i in range(len(level_nn)):
            if isinstance(level_nn[i], MaskedLinear):
                level_nn[i].set_mask(masks[mask_id])
                mask_id += 1
        level_nn = nn.Sequential(*level_nn)

        if self.verbose:
            print("hid_orderings", hid_orderings[0], "size", len(hid_orderings[0]))
            print("num hid layers:", len(hid_orderings))
            print(
                "expanded_input_ordering",
                expanded_input_ordering,
                "size",
                len(expanded_input_ordering),
            )
            print(
                "expanded_output_ordering",
                expanded_output_ordering,
                "size",
                len(expanded_output_ordering),
            )
        return level_nn, len(expanded_input_ordering), len(expanded_output_ordering)


### ======================================================================================

if __name__ == "__main__":
    # smoke test
    input_dim_dict = {"h": 2, "r": 1}
    hidden_sizes = [1, 2, 3]

    var_dim_dict = {"z1": 1, "y1": 1, "y2": 1, "x1": 1, "k": 1}
    dependency_dict = {
        "z1": set(["h", "r"]),
        "y1": set(["z1", "h"]),
        "y2": set(["y1", "z1", "h"]),
        "x1": set(["y1", "h"]),
        "k": set(["y1", "z1"]),
    }
    made = FullMade(input_dim_dict, hidden_sizes, dependency_dict, var_dim_dict)
    print("out_levels", made.out_levels)
    print("input_levels", made.input_levels)
    print(
        made._construct_level_ordering(
            made.input_levels[0], made.out_levels[0], dependency_dict
        )
    )
    print(
        made._construct_level_ordering(
            made.input_levels[1], made.out_levels[1], dependency_dict
        )
    )
    print(
        made._construct_level_ordering(
            made.input_levels[2], made.out_levels[2], dependency_dict
        )
    )
    inp = {"h": torch.tensor([1.0, 2.0]), "r": torch.tensor([0.5])}
    print(made(None, None, inp))
