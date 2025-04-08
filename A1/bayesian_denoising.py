import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from typing import Tuple

def create_mnist_sample(root_path: str, sample_size_training: int, sample_size_test: int, label: int = 0):
    training_dataset_path = os.path.join(root_path, 'sign_mnist_train.csv')
    test_dataset_path = os.path.join(root_path, 'sign_mnist_test.csv')

    training_dataset = pd.read_csv(training_dataset_path, header=None, low_memory=False)
    test_dataset = pd.read_csv(test_dataset_path, header=None, low_memory=False)

    training_sample = filter_dataset(training_dataset, label=label)
    test_sample = filter_dataset(test_dataset, label=label)

    training_sample = sample_from_dataset(training_sample, sample_size_training)
    test_sample = sample_from_dataset(test_sample, sample_size_test)

    return training_sample, test_sample

def filter_dataset(data: pd.DataFrame, label: int):
    mask = data.iloc[:, 0].values == str(label)
    index_list = data.index[mask]
    return data.loc[index_list]

def sample_from_dataset(data: pd.DataFrame, sample_size: int):
    sample_index_list = np.random.choice(data.index.tolist(), np.minimum(len(data), sample_size))
    return data.loc[sample_index_list].iloc[:, 1::]

def preprocess_data(training_sample: pd.DataFrame, test_sample: pd.DataFrame, noise_level: float):

    training_sample_processed = training_sample.values.copy().astype(np.float32)
    test_sample_processed = test_sample.values.copy().astype(np.float32)

    training_sample_processed = normalise_data(training_sample_processed)
    test_sample_processed = normalise_data(test_sample_processed)

    test_sample_processed = add_gaussian_noise(test_sample_processed, noise_level)

    return training_sample_processed, test_sample_processed

def normalise_data(image_data: np.ndarray) -> np.ndarray:
    return image_data / 255

def add_gaussian_noise(image_data: np.ndarray, noise_level: float) -> np.ndarray:
    return image_data + noise_level * np.random.randn(*image_data.shape)

def log_gaussian_kde(x: np.ndarray, data: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    this function computes the logarithm of approximate prior using kde for an array of input images x,
    the training data and the bandwidth.

    :param x: 2d numpy array of input data (i.e. arguments of the kde approximator), where each row
        corresponds to a flattened image
    :param data: 2d numpy array of training data which is used to construct the kde
    :param bandwidth: float
    :return: 2d numpy array containing for each input data item the kde approximator evaluated at x
    """
    pass

def log_likelihood(y: np.ndarray, x: np.ndarray, sigma: float) -> np.array:
    """
    this function computes the logarithm of the conditional density f_{Y|X=x}(y), i.e. the log-likelihood function
    of the denoising problem for an array of noisy input images y and an array of clean input images x

    :param y: 2d numpy array of noisy images, where each row of the array contains a flattened image
    :param x: 2d numpy array of clean images, where each row of the array contains a flattened image
    :param sigma: noise level
    :return: 2d array containing in row i, column j the log likelihood f_{Y|X=x[i, :]}(y[j, :])
    """
    pass

def visualise(denoised_image_index_arr: np.ndarray, training_data: np.ndarray, test_data: np.ndarray,
              image_shape: Tuple[int, int] = (28, 28)):
    num_plots_row = len(denoised_image_index_arr)

    fig, axs = plt.subplots(2, num_plots_row)
    for ax_col_idx, img_idx in enumerate(denoised_image_index_arr):
        axs[0, ax_col_idx].xaxis.set_visible(False)
        axs[0, ax_col_idx].yaxis.set_visible(False)
        axs[1, ax_col_idx].xaxis.set_visible(False)
        axs[1, ax_col_idx].yaxis.set_visible(False)

        axs[0, ax_col_idx].imshow(test_data[ax_col_idx, :].reshape(image_shape))
        axs[1, ax_col_idx].imshow(training_data[img_idx, :].reshape(image_shape))

    plt.show()


def main():
    data_root_path = 'fill me'          # path to directory containing 'sign_mnist_train.csv', 'sign_mnist_test.csv'
    training_sample_size = 'fill me'
    test_sample_size = 'fill me'
    noise_level = 'fill me'
    bandwidth = 'fill me'

    # load and preprocess data
    training_data, test_data = create_mnist_sample(root_path=data_root_path,
                                                   sample_size_training=training_sample_size,
                                                   sample_size_test=test_sample_size)
    training_data, test_data = preprocess_data(training_sample=training_data, test_sample=test_data,
                                               noise_level=noise_level)

    # compute log prior, log likelihood, log posterior
    llh = log_likelihood(test_data, training_data, noise_level)
    prior = log_gaussian_kde(training_data, training_data, bandwidth)
    posterior = llh + prior

    denoised_image_index_arr = np.argmax(posterior, axis=0)

    # generate plots
    visualise(denoised_image_index_arr, training_data=training_data, test_data=test_data)

if __name__ == '__main__':
    np.random.seed(123)
    main()