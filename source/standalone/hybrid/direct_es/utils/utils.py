# THIS FILE DEFINES SOME GENERAL ES BASED UTILITIES

from typing import Any
import torch


# defined because cannot use lambda with pytorch vmap etc
# modified from skrl.agents.torch.base.Agent._empty_preprocessor
def empty_preprocessor(input: Any, *args, **kwargs) -> Any:
    return input


"""ADD SOURCES AND LINKS TO THE BELOW"""


def _compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = torch.empty(len(x), dtype=torch.long, device=x.device)
    ranks[x.argsort()] = torch.arange(len(x), device=x.device)
    return ranks


def compute_centered_ranks(x):
    """
    Computes centered ranks in [-0.5, 0.5]
    """
    ranks = _compute_ranks(x.flatten()).reshape(x.shape).float()
    ranks /= x.numel() - 1
    ranks -= 0.5
    return ranks


def compute_weight_decay(weight_decay, model_param_tensor):
    """
    Computes the weight decay penalty for each agent.
    """
    # compute the mean of squared parameters for each agent
    mean_squared = torch.mean(model_param_tensor.pow(2), dim=1)  # Shape: [npop]

    weight_decay_penalty = -weight_decay * mean_squared  # Shape: [npop]

    return weight_decay_penalty


# taken from https://github.com/PaoloP84/EfficacyModernES/blob/master/modern_es.py#L511


def ascendent_sort(vect):
    # Create a copy of the vector
    tmpv = vect.clone()
    n = tmpv.size(0)
    # Index list to keep track of original indices
    index = torch.arange(n, dtype=torch.int32, device=vect.device)
    i = 0
    while i < n:
        # Look for minimum value in tmpv
        minv = tmpv[0]
        mini = 0
        j = 1
        while j < n:
            if tmpv[j] < minv:
                minv = tmpv[j]
                mini = j
            j += 1
        # Place the found minimum in the sorted array
        vect[i] = tmpv[mini]
        index[i] = mini
        i += 1
        # Set the selected minimum to a large value to prevent reuse
        tmpv[mini] = float("inf")

    return vect, index
