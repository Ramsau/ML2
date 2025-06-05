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

def denoising_score_matching_loss(model: EnergyGradientModel, x: torch.Tensor, sigma: float=0.05) -> torch.Tensor:
    """
    Function implementing the denoising score matching loss

    :param model: PyTorch module representing the gradient of the energy function
    :param x: PyTorch tensor representing the current training batch
    :param sigma: Noise level
    :return: Denoising score matching loss
    """
    z = torch.randn_like(x)
    y = x + (sigma * z)

    piecewise_activation = (-z / sigma) - model(y)
    loss = 0.5 * (piecewise_activation ** 2).mean()

    return loss


def train_ebm_denoising_score_matching(model: EnergyGradientModel, dataset: TensorDataset,
                                       max_num_iterations: int, batch_size: int=128) -> List[float]:
    """
    Function implementing the training loop of denoising score matching.

    :param model: PyTorch module representing the gradient of the energy to be trained
    :param dataset: Training dataset
    :param max_num_iterations: Maximal number of iterations to be performed
    :param batch_size: Size of training batches
    :return: List of training losses
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    k = 0
    stop = False
    while not stop:
        for batch in data_loader:
            k += 1
            if k > max_num_iterations:
                stop = True
                break

            loss = denoising_score_matching_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if k % 100 == 0:
                print(f"Iteration {k}, Loss: {loss.item()}")

    return loss_list

def main():
    device = torch.device('cpu')
    dtype = torch.float64

    num_data_samples = 2 ** 14
    data = load_data(num_data_samples, 'rings')
    dataset = create_training_dataset(data, device=device, dtype=dtype)

    visualise_training_dataset(dataset)

    # General hints
    # -------------
    #
    #   > small models do the job:
    #       * num_hidden_units <= 3
    #       * num_hidden_neurons <= 256
    #   > use standard activation functions such as torch.nn.ReLU
    #   > maximal number of iterations shouldn't be larger than 5000
    #   > use batch size in the range [4, 512]

    model = EnergyGradientModel(num_hidden_units=3,
                                num_hidden_neurons=256, activation_func=torch.nn.Sigmoid())
    model.to(dtype=dtype, device=device)

    max_num_iterations = 5000
    batch_size = 512
    loss_list = train_ebm_denoising_score_matching(model, dataset, max_num_iterations, batch_size)

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
