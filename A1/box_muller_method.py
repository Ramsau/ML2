import numpy as np
from matplotlib import pyplot as plt
from typing import Dict, Tuple

def box_muller_scheme(sample_size: int) -> np.ndarray:
    """
    function which creates a sample of size sample_size from the 2d-dimensional standard normal distribution
    using the box-muller method


    :param sample_size: size of random sample to be generated
    :return: 2d numpy array
    """

    # generate U,V ~ U(0,1)
    u = np.random.uniform(0, 1, sample_size)
    v = np.random.uniform(0, 1, sample_size)

    R = np.sqrt(-2 * np.log(u)) 
    W = 2 * np.pi * v 

    X = R * np.cos(W)
    Y = R * np.sin(W)

    return np.array([X, Y])  # shape (2, sample_size)


def marsaglia_bray_scheme(sample_size: int) -> np.ndarray:
    """
    function which implements the marsaglia-bray scheme to sample from 2d standard normal distribution

    :param sample_size: size of random sample to be generated
    :return: 2d numpy array
    """
    pass

def affine_linear_transformation(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    this function performs the affine linear transformation a * x + b for a 2x2 matrix a, an input vector
    b and the translation vector b

    :param x: 2d numpy array of shape (2, n), where n denotes the number of input data samples
    :param a: 2d numpy array representing the transformation matrix
    :param b: 2d numpy array of shape (2, 1) representing the translation vector
    :return: 2d numpy array of shape (2. n)
    """
    return np.matmul(a, x) + b

def marginal_max_likelihood_estimators(x: np.ndarray) -> Tuple[float, float]:
    """
    function which, given a sample of a 1d distribution, the maximum likelihood estimation
    of expected value and variance

    :param x: 1d numpy array
    :return: tuple of floats, where the first entry corresponds to the estimate of the expected value
        and the second entry corresponds to the estimate of the variance
    """
    mu_hat = np.mean(x)
    sig_hat_sq = np.mean((x - mu_hat) ** 2)
    return mu_hat, sig_hat_sq

def visualise(sample_array_dict: Dict[str, np.ndarray]):
    """
    this function creates a scatter plot of 2d random samples. each sample is
    visualised in its own plot

    :param sample_array_dict: dictionary, where each key (the name of the sample),
        the 2d array representing the corresponding sample is assigned
    :return:
    """

    
    pass

def main():
    sample_size = 'fill me'
    box_muller_sample = box_muller_scheme(sample_size)
    marsaglia_bray_sample = marsaglia_bray_scheme(sample_size)

    a1 = 'fill me'
    b1 = 'fill me'
    a2 = 'fill me'
    b2 = 'fill me'

    # transformations of box-muller sample
    transformed_sample_1 = affine_linear_transformation(box_muller_sample, a1, b1)
    transformed_sample_2 = affine_linear_transformation(box_muller_sample, a2, b2)

    # transformations of marsaglia-bray sample
    transformed_sample_3 = affine_linear_transformation(marsaglia_bray_sample, a1, b1)
    transformed_sample_4 = affine_linear_transformation(marsaglia_bray_sample, a2, b2)

    # maximum likelihood estimators of expected value and variance of the marginals
    for sample in [transformed_sample_1, transformed_sample_2,
                   transformed_sample_3, transformed_sample_4]:
        mu_hat_1, sig_hat_sq_1 = marginal_max_likelihood_estimators('fill me')
        mu_hat_2, sig_hat_sq_2 = marginal_max_likelihood_estimators('fill me')

        print('likelihood estimators of marginal expected value')
        print(mu_hat_1)
        print(mu_hat_2)

        print('likelihood estimators of marginal variance')
        print(sig_hat_sq_1)
        print(sig_hat_sq_2)

    # illustrations
    sample_array_dict = {}
    sample_array_dict['standard_normal'] = box_muller_sample
    sample_array_dict['normal_1'] = transformed_sample_1
    sample_array_dict['normal_2'] = transformed_sample_2
    visualise(sample_array_dict)

if __name__ == '__main__':
    main()