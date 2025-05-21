import torch
from torch.utils.data import IterableDataset, TensorDataset
import random
import numpy as np

class TrainingDataset(TensorDataset, IterableDataset):
    """
    Subclass of TensorDataset, IterableDataset used to model training datasets.
    """
    def __init__(self, data: torch.Tensor, dtype: torch.dtype, device: torch.device) -> None:
        IterableDataset.__init__(self)
        self._num_data_samples = data.shape[0]
        TensorDataset.__init__(self, data.to(dtype=dtype, device=device))

    def __iter__(self) -> torch.Tensor:
        while True:
            idx = random.choice(np.arange(0, self._num_data_samples))
            yield self.__getitem__(idx)
