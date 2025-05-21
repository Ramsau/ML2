import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.datasets import make_swiss_roll
from typing import List, Optional

from Dataset import TrainingDataset

def load_data(num_data_samples: int, dataset: str) -> Optional[np.ndarray]:
    """
    Function which loads data from named dataset.

    :param num_data_samples: Number of data samples to be generated
    :param dataset: Name of dataset
    :return: Numpy array of data samples of shape [batch_size, dimension].
    """
    data = None
    if dataset == 'swiss_roll':
        data, _ = make_swiss_roll(n_samples=num_data_samples, random_state=123, noise=0.5)
        data = data[:, [0, 2]]
        
        # normalise data - makes learning much easier!
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    elif dataset == 'rings':
        num_rings = 2
        num_ring_samples = num_data_samples // num_rings
        rad_array = (1 / num_rings) * np.arange(1, num_rings + 1)

        ring_array_list = []
        for i in range(0, num_rings):
            t = np.linspace(0, 2 * np.pi, num_ring_samples + 1)
            x = rad_array[i] * np.cos(t)
            y = rad_array[i] * np.sin(t)
            ring_array_list.append(np.array([x, y]))

        data = np.hstack(ring_array_list).transpose()
        data = data[np.random.randint(0, num_data_samples, size=(num_data_samples,))]
        data =  data + 0.03 * np.random.randn(data.shape[0], 2)
    elif dataset == 'gaussians':
        scale = 0.7
        rad = 4
        sig = 0.5

        tmp = 1 / np.sqrt(2)
        centers = [(1, 0), (-1, 0),
                   (0, 1), (0, -1),
                   (tmp, tmp), (-tmp, tmp),
                   (tmp, -tmp), (-tmp, -tmp)]
        centers = np.array([(rad * x, rad * y) for x, y in centers])

        data = sig * np.random.randn(num_data_samples, 2)
        data = scale * (data + centers[np.random.randint(len(centers), size=(num_data_samples,))])
    else:
        raise Exception('Unknown dataset.')

    return data

def create_training_dataset(data: np.ndarray, device: torch.device, dtype: torch.dtype) -> TensorDataset:
    """
    Function which creates a PyTorch TensorDataset

    :param data: Numpy array of data samples in shape [batch_size, dimension]
    :param device: PyTorch device - cpu, or cuda
    :param dtype: PyTorch datatype used to represent data of dataset
    :return:
    """
    return TrainingDataset(torch.from_numpy(data), dtype=dtype, device=device)

def collate_function(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    This function stacks all the data item sampled by the data loader into
    a single PyTorch tensor.

    :param batch: List of tuples of a single PyTorch tensor
    :return:
    """
    return torch.stack([item[0] for item in batch], dim=0)