import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple
import random

from DataUtils import load_data, create_training_dataset, collate_function
from VisualisationUtils import (visualise_energy_landscape, visualise_training_dataset,
                                visualise_samples, visualise_optimisation_stats)
from EnergyModel import EnergyModel
from SamplingUtils import sample

def max_likelihood_loss(model: EnergyModel, x: torch.Tensor,
                        x_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of the loss function

    :param model: PyTorch module representing the energy
    :param x: Current training batch
    :param x_0: Current initial guess for sampling
    :return: Tuple of loss and final Langevin sample
    """
    x_0 = x_0.detach()  # Ensure x_0 is detached from the graph
    gradient_func = lambda z: torch.autograd.grad(outputs=torch.sum(model(z)), inputs=z)[0]
    langevin_sample = sample(
        gradient_func=gradient_func,
        x_0=x_0, num_sampling_steps=20)
    
    x_hat = langevin_sample[-1]  # Take the last sample from the Langevin chain

    energy_x = model(x)
    energy_x_hat = model(x_hat)


    # Compute the loss as the difference in energies
    loss = -torch.mean(energy_x_hat - energy_x)
    return loss, x_hat

def train_ebm_max_likelihood(model: EnergyModel, dataset: TensorDataset,
                             max_num_iterations: int, batch_size: int=128, device="cpu", dtype=torch.float64) -> List[float]:
    """
    Main training routine for the training of an EBM using the maximum likelihood approach.

    NOTES
    -----
        > The initial guess for the sample generating function does not need to be generated each time
            by means of torch.rand(), or torch.randn()
        > It is convenient to use previous iterates as initial guesses.

    :param model: PyTorch module representing the energy function of the EBM
    :param dataset: Training dataset
    :param max_num_iterations: Maximal number of iterations
    :param batch_size: Batch size
    :return: List of training losses
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)
    x_0 = torch.randn(batch_size, 2, dtype=dataset.tensors[0].dtype, device=dataset.tensors[0].device)
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    # print(data_loader.dataset[0].shape)
    k = 0
    stop = False
    while not stop:
        for batch in data_loader:
            k+= 1
            if k > max_num_iterations:
                stop = True
                break
            loss, x_0 = max_likelihood_loss(model, batch, x_0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if k % 100 == 0:
                print(f"Iteration {k}, Loss: {loss.item()}")

        break
    return loss_list

def main():
    device = torch.device('cuda:2')
    dtype = torch.float64

    print(f"Using device: {device}")

    num_data_samples = 2 ** 14
    data = load_data(num_data_samples, 'swiss_roll')
    dataset = create_training_dataset(data, device, dtype)

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


    energy_model = EnergyModel(num_hidden_units=2,
                               num_hidden_neurons=128, activation_func=torch.nn.ReLU())
    energy_model.to(dtype=dtype, device=device)

    max_num_iterations = 3000
    batch_size = 512
    loss_list = train_ebm_max_likelihood(energy_model, dataset, max_num_iterations, batch_size=batch_size, device=device, dtype=dtype)
    x_0 = torch.randn(3000, 2).to(dtype=dtype, device=device)

    gradient_func = lambda z: torch.autograd.grad(outputs=torch.sum(energy_model(z)), inputs=z)[0]
    sample_list = sample(gradient_func, x_0, step_size=5e-4, num_sampling_steps=1000)
    # move model back to cpu for visualisation
    energy_model.to(device='cpu', dtype=torch.float64)
    a = 3.0
    visualise_energy_landscape(energy_model, x_box_low=-a, x_box_high=a,
                               y_box_low=-a, y_box_high=a, num_samples=200, dtype=dtype)
    visualise_samples(torch.cat([sample_list[-1]]))
    visualise_optimisation_stats(loss_list)

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    random.seed(123)
    main()