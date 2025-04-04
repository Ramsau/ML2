import numpy as np
from matplotlib import pyplot as plt
from typing import List

def gaussian_kde(x: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    this function computes the gaussian-kde given an array of input data x, the data sample and
    a bandwidth

    :param x: 1d numpy array representing the input data
    :param data: 1d array representing the data sample
    :param bandwidth: float, larger than zero
    :return: 1d numpy array of function values of kde evaluated at x
    """
    return np.sum((1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp((-0.5 / bandwidth ** 2) *
                                                    (x.reshape(-1, 1) - data.reshape(1, -1)) ** 2), axis=1)

def visualise_kde(x_arr: np.ndarray, y_arr_list: List[np.ndarray], data_sample: np.ndarray,
                  bandwidth_list: List[float]):
    """
    function which visualises different kdes.

    :param x_arr: 1d numpy array of inputs used to compute the function values of each kde
    :param y_arr_list: list of 1d numpy arrays containing the function values of each kde w.r.t to x
    :param data_sample: 1d array representing the data sample which was used to create the kde
    :param bandwidth_list: list of floats containing the bandwidths
    :return: /
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_sample, np.zeros(len(data_sample)), marker='x')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    for i in range(0, len(y_arr_list)):
        ax.plot(x_arr, y_arr_list[i], label='{:.3f}'.format(bandwidth_list[i]))
    fig.legend()
    plt.show()

def main():
    data_sample = np.array([0.1, 0.3, 0.4, 0.95, 1.1, 1.33, 1.4])

    data_min = np.min(data_sample)
    data_max = np.max(data_sample)
    x_arr = x = np.linspace(int(data_min) - 1, int(data_max) + 1, 1001)

    bandwidth_list = [0.03, 0.1, 0.3, 1]
    y_arr_list = [gaussian_kde(x, data_sample, bandwidth=h) for h in bandwidth_list]

    visualise_kde(x_arr, y_arr_list, data_sample, bandwidth_list)

if __name__ == '__main__':
    main()