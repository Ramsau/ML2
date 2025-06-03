import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
from typing import List

from DataUtils import create_training_dataset, collate_function, load_data
from EnergyModel import EnergyGradientModel
from VisualisationUtils import (visualise_optimisation_stats, visualise_training_dataset,
                                visualise_samples, visualise_gradient_field)
from SamplingUtils import sample

def implicit_score_matching_loss(model: EnergyGradientModel, x: torch.Tensor) -> torch.Tensor:
    """
    Function implementing the implicit score matching loss

    :param model: PyTorch module representing the gradient of the EBM to be trained.
    :param x: Current training batch
    :return: Implicit score matching loss
    """
    x.requires_grad_(True)
    model.train()
    score = model.forward(x)
    norm = torch.norm(score, 2, dim=1) / 2.0
    score_grad = torch.autograd.grad(inputs=x, outputs=score.sum(), grad_outputs=torch.ones_like(score))[0]
    trace = score_grad.trace()
    pass

def train_ebm_implicit_score_matching(model: EnergyGradientModel, dataset: TensorDataset,
                                      max_num_iterations: int, batch_size: int=512) -> List[float]:
    """
    Function implementing the training loop of implicit score matching.

    :param model: PyTorch module representing the model to be trained.
    :param dataset: Training dataset
    :param max_num_iterations: Maximal number of iterations
    :param batch_size: Size of training batches
    :return: List of training losses
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []

    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)
    k = 0
    stop = False
    while not stop:
        for batch in data_loader:
            optimiser.zero_grad()
            loss = implicit_score_matching_loss(model, batch)
            loss_list.append(loss.detach().cpu().item())

            loss.backward()
            optimiser.step()

            if (k + 1) % 100 == 0:
                print('iteration [{:d}/{:d}]: loss = {:.5f}'.format(k + 1, max_num_iterations, loss))

            if (k + 1) == max_num_iterations:
                print('reached maximal number of iterations')
                stop = True
                break
            else:
                k += 1

    return loss_list

def main():
    device = torch.device('cpu')
    dtype = torch.float64

    num_data_samples = 2 ** 10
    data = load_data(num_data_samples, 'rings')
    dataset = create_training_dataset(data, device=device, dtype=dtype)

    visualise_training_dataset(dataset)

    model = EnergyGradientModel(activation_func=torch.nn.Softplus(), num_hidden_neurons=512, num_hidden_units=2)
    model.to(dtype=dtype, device=device)

    max_num_iterations = 5000
    loss_list = train_ebm_implicit_score_matching(model, dataset, max_num_iterations)

    visualise_optimisation_stats(loss_list)

    a = 1.5
    visualise_gradient_field(model, x_box_low=-a, x_box_high=a, y_box_low=-a, y_box_high=a,
                             num_samples=100, dtype=dtype)

    num_chains = 1000
    x0 = 2 * torch.rand(num_chains, 2) - 1
    x0 = x0.to(dtype=dtype, device=device)
    samples = sample(lambda z: -model.forward(z), x0, step_size=5e-4, num_sampling_steps=1000)
    visualise_samples(torch.cat([samples[-1]]))

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)
    main()
