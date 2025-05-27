import torch
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List
import numpy as np

def visualise_energy_landscape(model: torch.nn.Module, x_box_low: float, x_box_high: float,
                               y_box_low: float, y_box_high: float, num_samples: int, dtype: torch.dtype) -> None:
    """
    Function generating the energy landscape given the energy model

    :param model: PyTorch module representing the energy
    :param x_box_low: Lower bound of x range
    :param x_box_high: Upper bound of x range
    :param y_box_low: Lower bound of y range
    :param y_box_high: Upper bound of y range
    :param num_samples: Number of samples used to generate partition x and y range respectively
    :param dtype: PyTorch datatype
    :return:
    """
    x = torch.linspace(x_box_low, x_box_high, num_samples)
    y = torch.linspace(y_box_low, y_box_high, num_samples)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    points = points.to(dtype=dtype)
    ee = model(points).reshape(num_samples, num_samples)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    cmsh =ax.pcolormesh(xx.detach().cpu().numpy(), yy.detach().cpu().numpy(), ee.detach().cpu().numpy(),
                            cmap=plt.cm.inferno)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(cmsh, cax=cax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_box_low, x_box_high)
    ax.set_ylim(y_box_low, y_box_high)
    ax.set_title('energy landscape')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
    # savefig
    plt.savefig('energy_landscape.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualise_training_dataset(dataset: TensorDataset) -> None:
    """
    Function which creates visualisation of the training dataset by means of
    a scatter plot

    :param dataset: Training dataset in terms of TensorDataset
    :return:
    """
    data_array = dataset.tensors[0].detach().cpu().numpy()

    x_box_low = np.min(data_array[:, 0]) - 1
    x_box_high = np.max(data_array[:, 0]) + 1
    y_box_low = np.min(data_array[:, 1]) - 1
    y_box_high = np.max(data_array[:, 1]) + 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_array[:, 0], data_array[:, 1], s=0.8)
    ax.set_title('training dataset')
    ax.set_xlim([x_box_low, x_box_high])
    ax.set_ylim([y_box_low, y_box_high])
    ax.set_aspect('equal', adjustable='box')
    # savefig
    plt.savefig('training_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def compute_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Function which computes the moving average of a data array.

    :param data: Data array whose moving average needs to be computed
    :param window: Integer indicating the number of data points which is used to
        compute the average
    :return:
    """
    lst = [np.nan for _ in range(0, window)]
    for i in range(0, len(data) - window + 1):
        lst.append(np.mean(data[i : i + window]))
    return np.array(lst)

def visualise_optimisation_stats(loss_list: List[float]) -> None:
    """
    Function to visualise the training loss

    :param loss_list: List of training losses
    :return:
    """
    moving_average = compute_moving_average(np.array(loss_list), 10)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(0, len(loss_list)), loss_list, label='training loss')
    ax.plot(np.arange(1, len(moving_average) + 1), moving_average, color='orange',
            label='moving average of training loss')
    ax.set_title('training loss')
    ax.legend()

    plt.show()
    # savefig
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualise_samples(sample: torch.Tensor) -> None:
    """
    Function for the visualisation of a data sample by means of a scatter plot

    :param sample: PyTorch tensor containing the data items to be depicted; this tensor
        is assumed to be of shape [batch_size, 2]
    :return:
    """
    sample_array = sample.detach().cpu().numpy()

    x_box_low = np.min(sample_array[:, 0]) - 1
    x_box_high = np.max(sample_array[:, 0]) + 1
    y_box_low = np.min(sample_array[:, 1]) - 1
    y_box_high = np.max(sample_array[:, 1]) + 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(sample_array[:, 0], sample_array[:, 1], s=0.8)
    ax.set_title('generated samples')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([x_box_low, x_box_high])
    ax.set_ylim([y_box_low, y_box_high])

    plt.show()
    # savefig
    plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def visualise_gradient_field(model: torch.nn.Module, x_box_low: float, x_box_high: float,
                             y_box_low: float, y_box_high: float, num_samples: int, dtype: torch.dtype) -> None:
    """
    Function which is used to visalise a vector field using plt.streamplot(...)

    :param model: PyTorch representing the gradient to be visualised
    :param x_box_low: Lower bound of x range
    :param x_box_high: Upper bound of x range
    :param y_box_low: Lower bound of y range
    :param y_box_high: Upper bound of y range
    :param num_samples: Number of samples used to generate partition x and y range respectively
    :param dtype: PyTorch datatype
    :return:
    """

    x = np.linspace(x_box_low, x_box_high, num_samples)
    y = np.linspace(y_box_low, y_box_high, num_samples)
    xx, yy = np.meshgrid(x, y)

    zz = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = torch.tensor(zz, dtype=dtype)
    outputs = model(zz).detach().numpy()
    u = outputs[:, 0].reshape(xx.shape)
    v = outputs[:, 1].reshape(yy.shape)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.streamplot(xx, yy, u, v)
    ax.set_xlim([x_box_low, x_box_high])
    ax.set_ylim([y_box_low, y_box_high])

    plt.show()
    plt.close(fig)
