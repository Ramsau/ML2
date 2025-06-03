import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List
import numpy as np
import random

from SamplingUtils import sample
from DataUtils import create_training_dataset, collate_function, load_data
from VisualisationUtils import (visualise_energy_landscape, visualise_training_dataset,
                                visualise_optimisation_stats, visualise_samples)
from EnergyModel import NormalisedEnergyModel

def nce_loss(energy_model: NormalisedEnergyModel, noise_model: torch.distributions.Distribution,
             x: torch.Tensor) -> torch.Tensor:
    """
    Function which computes the loss for noise contrastive estimation.

    :param energy_model: Module representing the energy function
    :param noise_model: Noise distribution
    :param x: Tensor representing the current training data batch
    :return: NCE loss
    """
    noise = noise_model.sample(x.shape[:1]).to(dtype=x.dtype, device=x.device)

    log_p_x = -energy_model(x).flatten()
    log_q_x = noise_model.log_prob(x)
    log_p_y = -energy_model(noise).flatten()
    log_q_y = noise_model.log_prob(noise)
    # log(a/b) = log(a) - log(b)
    loss_x = log_p_x - torch.log(torch.exp(log_p_x) + torch.exp(log_q_x))
    loss_y = log_q_y - torch.log(torch.exp(log_p_y) + torch.exp(log_q_y))
    return -loss_x.sum() - loss_y.sum()

def train_ebm_noise_contrastive_estimation(energy_model: NormalisedEnergyModel,
                                           noise_model: torch.distributions.Distribution,
                                           dataset: TensorDataset, max_num_iterations: int,
                                           batch_size: int=128) -> List[float]:
    """
    Function which incorporates the NCE training routine.

    :param energy_model: PyTorch module representing the energy function of the EBM to be fitted
    :param noise_model: Noise model used to generate fake/noisy data
    :param dataset: torch dataset representing the training dataset
    :param max_num_iterations: Maximal number of iterations to be performed
    :param batch_size: Training batch size
    :return: List of training losses
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_function)
    loss_list = []
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=1e-3)
    energy_model.train()

    k = 0
    stop = False
    while not stop:
        for batch in data_loader:
            k += 1
            if k > max_num_iterations:
                stop = True
                break

            loss = nce_loss(energy_model, noise_model, batch)
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
    data = load_data(num_data_samples, 'gaussians')
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

    energy_model = NormalisedEnergyModel(num_hidden_units=3,
                                         num_hidden_neurons=256, activation_func=torch.nn.LeakyReLU())
    energy_model.to(dtype=dtype, device=device)
    noise_model = torch.distributions.MultivariateNormal(torch.zeros(2).to(device), 4 * torch.eye(2).to(device))

    max_num_iterations = 1000
    batch_size = 100
    loss_list = train_ebm_noise_contrastive_estimation(energy_model, noise_model, dataset,
                                                       max_num_iterations, batch_size=batch_size)

    a = 5.0
    visualise_energy_landscape(energy_model, x_box_low=-a, x_box_high=a, y_box_low=-a, y_box_high=a,
                               num_samples=100, dtype=dtype)
    visualise_optimisation_stats(loss_list)

    gradient_func = lambda z: torch.autograd.grad(outputs=torch.sum(energy_model(z)), inputs=z)[0]

    x_0 = 2 * (2 * torch.rand(1000, 2).to(dtype=dtype, device=device) - 1)
    sample_list = sample(gradient_func, x_0, step_size=5e-4, num_sampling_steps=1000)
    visualise_samples(torch.cat([sample_list[-1]]))

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    main()
