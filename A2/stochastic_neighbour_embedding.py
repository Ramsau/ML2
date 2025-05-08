from typing import Optional, List, Tuple, Callable
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os

EPS = 1e-10

def load_mnist_data(root_path: str, num_training_samples: int = -1, num_test_samples: int = -1,
                    class_list: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    training_dataset_path = os.path.join(root_path, 'mnist_train.csv')
    test_dataset_path = os.path.join(root_path, 'mnist_test.csv')

    training_dataset = pd.read_csv(training_dataset_path, header=None)
    training_dataset_filtered = filter_mnist_data(training_dataset, class_list)

    test_dataset = pd.read_csv(test_dataset_path, header=None)
    test_dataset_filtered = filter_mnist_data(test_dataset, class_list)

    training_sample = sample_from_mnist_data(training_dataset_filtered, num_training_samples)
    test_sample = sample_from_mnist_data(test_dataset_filtered, num_test_samples)

    return training_sample, test_sample

def filter_mnist_data(data: pd.DataFrame, class_list: List[int]) -> pd.DataFrame:
    classes = np.arange(0, 10, dtype=np.uint) if class_list is None else class_list
    mask = [label in classes for label in data.iloc[:, 0]]
    index_list = data.index[mask]
    data_filtered = data.loc[index_list]

    return data_filtered

def sample_from_mnist_data(data: pd.DataFrame, sample_size: int = -1) -> pd.DataFrame:
    size = sample_size if sample_size > 0 else len(data)
    sample_index_list = np.random.choice(data.index.tolist(), size)
    return data.loc[sample_index_list]

def preprocess_mnist_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    image_labels = data.iloc[:, 0]
    image_data = data.iloc[:, 1::]
    image_arrays = normalise_data(image_data.values)
    return image_arrays, image_labels.values

def normalise_data(data: np.ndarray) -> np.ndarray:
    return 2 * (data / 255) - 1

def binary_search(func: Callable, target_value: float, abs_tol: float = 1e-6, lower: float = 0,
                  upper: float = 1e4, max_num_iterations: int = 250) -> float:
    """
    this function applies a binary search to determine a root of |func(...) - target_value|

    :param func: function on which binary search is applied
    :param target_value: value to be reached by the function func
    :param abs_tol: absolute tolerance used as stopping criterion
    :param lower: initial lower bound of binary search algorithm
    :param upper: initial upper bound of binary search algorithm
    :param max_num_iterations: maximal number of iterations
    :return: approximate root of |func(...) - target_value|
    """
    a = lower
    b = upper
    x = 0.5 * (a + b)
    for k in range(0, max_num_iterations):
        y = func(x)
        if y > target_value:
            b = x
        else:
            a = x

        if np.abs(y - target_value) < abs_tol:
            break
        x = 0.5 * (a + b)
    return x

def compute_perplexity(similarities: np.ndarray) -> float:
    """
    this function computes the perplexity for a given array of similarities

    :param similarities: 1d numpy array of similarities
    :return: perplexity
    """
    return 2 ** (-np.sum(similarities * np.log2(similarities + EPS)))

def compute_standard_deviation_by_perplexity(target_perplexity: float, distance_matrix: np.ndarray) -> np.ndarray:
    """
    this function is used to determine the parameters sigma_1, ..., sigma_n such that the target perplexity
    is reached.

    :param target_perplexity: perplexity to be reached along the high-dimensional dataset
    :param distance_matrix: 2d numpy matrix of pairwise distances in high-dimensional space
    :return: 1d array containing the sought parameters
    """
    num_samples = distance_matrix.shape[0]
    sig_array = np.zeros(num_samples)
    for i in range(0, num_samples):
        distances_i = distance_matrix[i, :]

        def objective_func(sig: float) -> float:
            """
            this function computes the perplexity of the i-th row of the distance matrix; it is
            used to determine the target complexity.

            :param sig: float to be fitted such that target perplexity is met.
            :return: perplexity of distances_i, given the parameter sig
            """
            tmp = -0.5 * distances_i / sig ** 2
            exp_terms = np.exp(tmp - np.max(tmp))
            exp_terms[i] = 0
            sim = exp_terms / np.sum(exp_terms)
            return compute_perplexity(sim)

        sigma = binary_search(objective_func, target_value=target_perplexity)
        sig_array[i] = sigma
    return sig_array

def compute_distance_matrix(data: np.ndarray) -> np.ndarray:
    """
    this function computes the pairwise squared euclidian distances of the data points.

    :param data: numpy data array, where i-th row corresponds to the i-th data item
    :return: 2d numpy array of distances, where the item in row i, column j equals the squared euclidian norm
        of the difference of the i-th and j-th data item
    """
    return data[:, np.newaxis, :] - data[np.newaxis, :, :]

def compute_high_dim_similarity_matrix(distance_matrix: np.ndarray, sigma_array: np.ndarray) -> np.ndarray:
    """
    this function computes the similarities p_{j|i} given the squared distances and the parameters sigma_i

    :param distance_matrix: 2d numpy array of squared distances of pairwise difference of data points
    :param sigma_array: 1d numpy array of parameters sigma_i required for the computation of similarities
    :return: 2d numpy array where row i, column j contains the similarity measure p_{j|i}
    """
    pass

def symmetrise_similarity_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    this function is used to symmetrise the similarity matrix (p_{j|i})

    :param similarity_matrix: 2d numpy array of the similarities p_{j|i}
    :return: 2d numpy array
    """
    num_samples = similarity_matrix.shape[0]
    return (similarity_matrix + np.transpose(similarity_matrix)) / (2 * num_samples)

def compute_low_dim_similarity_matrix_tsne(distance_matrix: np.ndarray) -> np.ndarray:
    """
    this function computes the similarity matrix in low-dimensional space given the distance matrix
    of the low-dimensional representatives

    :param distance_matrix: 2d numpy array
    :return: 2d numpy array of similarity measures q_{i, j}
    """
    tmp = 1.0 / (1.0 + distance_matrix)
    np.fill_diagonal(tmp, 0.0)
    sim_matrix = tmp / np.sum(tmp)
    return sim_matrix

def compute_gradient_tsne(y: np.ndarray, similarity_matrix_high_dim: np.ndarray,
                          similarity_matrix_low_dim: np.ndarray) -> np.ndarray:
    """
    this function computes the gradient of the kl-divergence between the similarity measures in high- and low-
    dimensional space

    :param y: 2d numpy array of shape (num_training_samples, 2) containing the low-dimensional representatives
    :param similarity_matrix_high_dim: 2d array of similarities p_{i, j}
    :param similarity_matrix_low_dim: 2d array of similarities q_{i, j}
    :return: gradient of kl-divergence between high- and low-dimensional similarity distributions
    """
    pass

def initial_guess(sample_size: int, eps: float = 1e-4) -> np.ndarray:
    """
    this function provides an initial guess for the representatives of the data in
    low-dimensional space

    :param sample_size: number of data samples
    :param eps: variance of 2d normal is used for initialisation
    :return: 2d numpy array of samples from 2d-normal
    """
    mu = np.zeros(2)
    sigma = eps * np.eye(2)
    return np.random.multivariate_normal(mu, sigma, sample_size)

def compute_loss(similarities_high_dim: np.ndarray, similarities_low_dim: np.ndarray) -> float:
    """
    this function computes the loss of the optimisation task, which in this case corresponds
    to the kl-divergence between the similarities in high- and low-dimensional space.

    :param similarities_high_dim: 2d array of similarities in high-dimensional space
    :param similarities_low_dim: 2d array of similarities in low-dimensional space
    :return: float
    """
    return np.sum(similarities_high_dim * np.log(similarities_high_dim / similarities_low_dim))

def train_tsne(data: np.ndarray, num_iterations: int = 500, perplexity: float = 20, exaggeration: float = 4,
               exaggeration_iter_thresh: int = 50) -> np.ndarray:
    """
    this function incorporates the tsne training loop.

    :param data: 2d numpy array respresenting the high-dimensional data, where data.shape[0] corresponds
        to the number of data samples, and data.shape[1] the dimensionality of the data
    :param num_iterations: number of iterations (i.e. gradient steps)
    :param perplexity:
    :param exaggeration: factor with which the high-dimensional similarity measures are multiplied in
        the first states of the training loop
    :param exaggeration_iter_thresh: iteration beginning with which exxageration is omitted
    :return: 2d numpy array of low-dimensional representatives of high-dimensional data.
    """
    alpha = 500
    beta = 0.7

    y0 = initial_guess(data.shape[0])

    y_old = y0.copy()
    y = y0.copy()

    distance_matrix_high_dim = compute_distance_matrix(data)
    sigma_array = compute_standard_deviation_by_perplexity(perplexity, distance_matrix_high_dim)
    similarities_high_dim = compute_high_dim_similarity_matrix(distance_matrix_high_dim, sigma_array)
    similarities_high_dim = symmetrise_similarity_matrix(similarities_high_dim)
    similarities_high_dim = exaggeration * similarities_high_dim

    similarities_high_dim = np.maximum(similarities_high_dim, 1e-10)

    for k in range(0, num_iterations):

        'HEAVY BALL UPDATE HERE'

        if (k + 2) == exaggeration_iter_thresh:
            # renounce on exaggeration after some time
            similarities_high_dim = similarities_high_dim / exaggeration

        if (k + 1) % 10 == 0:
            loss = compute_loss(similarities_high_dim, similarities_low_dim)
            print('iteration [{:d}/{:d}]: loss = {:.7f}'.format(k + 1, num_iterations, loss))

    return y

def visualise(low_dim_data: np.ndarray, image_labels: np.ndarray) -> None:
    """
    this function visualises the result of the stochastic neighbour embedding by means of scatter plots.

    :param low_dim_data: 2d array of low-dimensional representatives of the data, where i-th row contains
        the representative of the i-th data item
    :param image_labels: 1d array containing the labels of the original data
    :return: /
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = matplotlib.colormaps['tab10'].colors

    class_list = list(set(image_labels))
    for i, item in enumerate(class_list):
        idx_subset_mask = (image_labels == item)
        subset = low_dim_data[idx_subset_mask, :]
        ax.scatter(subset[:, 0], subset[:, 1], color=colors[i], label=str(item), s=4)
    fig.legend()
    plt.show()

def main():
    data_root_path = '.'             # path to directory containing 'sign_mnist_train.csv', 'sign_mnist_test.csv'

    num_training_samples = 100
    num_iterations = 500
    perplexity = 20
    class_list = [0, 1, 2, 3, 4]

    # load data
    data, _ = load_mnist_data(root_path=data_root_path, num_training_samples=num_training_samples,
                              num_test_samples=0, class_list=class_list)
    image_data, image_labels = preprocess_mnist_data(data)

    # train
    low_dim_data = train_tsne(image_data, num_iterations=num_iterations, perplexity=perplexity)

    # visualise
    visualise(low_dim_data, image_labels)

if __name__ == '__main__':
    np.random.seed(123)
    main()
