import torch
import numpy as np
from typing import List, Callable

def sample(gradient_func: Callable, x_0: torch.Tensor, step_size: float=5e-4,
           num_sampling_steps: int = 100) -> List[torch.Tensor]:
    """
    Function which samples from the distribution induced by the current EBM by means of
    ULA.

    :param gradient_func: Implementation of the gradient of a function; it takes PyTorch tensors
        as input and returns PyTorch tensors as output.
    :param x_0: initial sampling guess of shape (batch_size, 2). If batch_size > 1,
        multiple Langevin shall be generated
    :param step_size: Langevin step size
    :param num_sampling_steps: Number of Langevin steps to be performed.
    :return: List of generated samples.
    """
    pass