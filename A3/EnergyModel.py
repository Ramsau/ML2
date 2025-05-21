import torch

class CoreEnergyModel(torch.nn.Module):
    """
    Base class of EBM models. It consists of a block of hidden layers -
    the output layer needs to be specified in respective subclasses.
    """

    def __init__(self, input_dim: int, num_hidden_units: int,
                 num_hidden_neurons: int, activation_func: torch.nn.Module) -> None:
        """
        Initialisation of base class. Note that, all the layers are per construction of the
        same size and have the same activation function.

        :param input_dim: Input dimension
        :param num_hidden_units: Number of hidden layers
        :param num_hidden_neurons: Number of neurons per hidden layer
        :param activation_func: Activation function per layer
        """
        super().__init__()

        self._input_dim = 2
        self._num_hidden_neurons = num_hidden_neurons
        self._num_hidden_units = num_hidden_units

        self._input_layer = torch.nn.Linear(in_features=input_dim, out_features=self._num_hidden_neurons)
        self._hidden_block = torch.nn.Sequential(*[torch.nn.Linear(in_features=self._num_hidden_neurons,
                                                                  out_features=self._num_hidden_neurons)
                                                   for _ in range(0, self._num_hidden_units)])
        self._activation_func = activation_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._activation_func(self._input_layer(x))
        for item in self._hidden_block:
            y = self._activation_func(item(y))
        y = self._output_layer(y)
        return y

class EnergyModel(CoreEnergyModel):
    """
    Subclass of CoreEnergyModel modelling the energy function of an EBM. Note that the output dimension of this
    module equals to 1.
    """
    def __init__(self, input_dim: int=2, num_hidden_units: int=2, num_hidden_neurons: int=128,
                 activation_func: torch.nn.Module = torch.nn.ReLU()):
        super().__init__(input_dim, num_hidden_units, num_hidden_neurons, activation_func)
        self._output_layer = torch.nn.Linear(in_features=num_hidden_neurons, out_features=1)

class NormalisedEnergyModel(CoreEnergyModel):
    """
    Subclass of CoreEnergyModel modelling the energy function of an EBM while considering its normalising
    constant as trainable parameter. The output dimension of the module equals 1.
    """
    def __init__(self, input_dim: int=2, num_hidden_units: int=2, num_hidden_neurons: int=128,
                 activation_func: torch.nn.Module = torch.nn.ReLU()):
        super().__init__(input_dim, num_hidden_units, num_hidden_neurons, activation_func)

        self.log_inv_z = torch.nn.Parameter(2 * torch.rand(1) - 1, requires_grad=True)
        self._output_layer = torch.nn.Linear(in_features=num_hidden_neurons, out_features=1)

    def get_log_inv_normalising_const(self) -> torch.Tensor:
        return self.log_inv_z.data

class EnergyGradientModel(CoreEnergyModel):
    """
    Subclass of CoreEnergyModel used to model score of a 2d EBM. Thus, the output dimension of this network
    equals 2.
    """
    def __init__(self, input_dim: int=2, num_hidden_units: int=2, num_hidden_neurons: int=128,
                 activation_func: torch.nn.Module = torch.nn.Softplus()):
        super().__init__(input_dim, num_hidden_units, num_hidden_neurons, activation_func)
        self._output_layer = torch.nn.Linear(in_features=num_hidden_neurons, out_features=2)
