import numpy as np
from matplotlib import pyplot as plt

from langevin_sampling_quadratic import mala

def energy_func_double_banana(x1: np.ndarray, x2: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    this function computes the energy of the double-banana distribution

    :param x1: numpy array of values of x-coordinate
    :param x2: numpy array of values of y-coordinate
    :param a: non-negative parameter value
    :param b: non-negative parameter value
    :return: numpy array of function values
    """
    pass

def grad_energy_func_double_banana(x1: np.ndarray, x2: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    this function computes the gradient of the energy function of the double banana distribution

    :param x1: numpy array of values of x-coordinate
    :param x2: numpy array of values of y-coordinate
    :param a: non-negative parameter value
    :param b: non-negative parameter value
    :return: numpy array containing partial derivative w.r.t the first coordinate and partial derivative w.r.t. the
        second coordinate
    """
    pass

def density_func_double_banana(x1: np.ndarray, x2: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    this function computes the density of the double-banana distribution

    :param x1: numpy array of values of x-coordinate
    :param x2: numpy array of values of y-coordinate
    :param a: non-negative parameter value
    :param b: non-negative parameter value
    :return: numpy array of densities at (x1, x2)
    """
    return np.exp(-energy_func_double_banana(x1, x2, a, b))

def visualise(sample: np.ndarray, a: float, b: float):
    """
    this function visualises the exact density of the double-banana distribution (contour plot) and
    a sample of it (scatter plot)

    :param sample: 2d numpy array of shape (num_samples, 2)
    :param a: non-negative parameter value
    :param b: non-negative parameter value
    :return: /
    """
    x = np.linspace(-5, 5, 101)
    y = np.linspace(-5, 5, 101)
    xx, yy = np.meshgrid(x, y)

    zz = density_func_double_banana(xx, yy, a, b)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.pcolormesh(xx, yy, zz, cmap='viridis')
    ax.scatter(sample[:, 0], sample[:, 1], marker='x', s=0.8, color='black')

    plt.show()
    plt.close(fig)

def main():
    a = 0.6
    b = 1.8

    num_iterations = 'fill me'
    num_chains = 'fill me'
    gamma = 'fill me'
    x0 = 2 * (np.random.rand(2, num_chains) - 1)
    burn_in = 3000

    langevin_sample = mala(lambda x: grad_energy_func_double_banana(x[0], x[1], a, b),
                           lambda x: density_func_double_banana(x[0], x[1], a, b), num_iterations, gamma, x0)
    langevin_sample = np.hstack(langevin_sample[burn_in::, :, :]).transpose()

    visualise(langevin_sample, a, b)

if __name__ == '__main__':
    np.random.seed(123)
    main()





