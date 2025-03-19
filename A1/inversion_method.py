from typing import Tuple, Any, Dict
import numpy as np
from matplotlib import pyplot as plt

def pdf_f(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    implementation of pdf f

    :param x: 1d numpy array of input values in the range of (-inf, inf)
    :param a: value of parameter a
    :param b: value of parameter b
    :return: 1d numpy array of function values
    """
    pass

def pdf_g(x: np.ndarray, lam: float) -> np.ndarray:
    """
    implementation of pdf g

    :param x: 1d numpy array of input values in the range of (-inf, inf)
    :param lam: value of parameter lambda
    :return: 1d numpy array of function values
    """
    pass

def pdf_h(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    implementation of pdf h

    :param x: 1d numpy array of input values in the range of (-inf, inf)
    :param alpha: value of parameter alpha
    :param beta: value of parameter beta
    :return: 1d numpy array of function values
    """
    pass

def inverse_cdf_f(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    generalised inverse of cdf of f

    :param y: 1d numpy array of input values in the range of [0, 1]
    :param a: value of parameter a
    :param b: value of parameter b
    :return: 1d numpy array of function values
    """
    pass

def inverse_cdf_g(y: np.ndarray, lam: float) -> np.ndarray:
    """
    generalised inverse of cdf of g

    :param y: 1d numpy array of input values in the range of [0, 1]
    :param lam: value of parameter lambda
    :return: 1d numpy array of function values
    """
    pass

def inverse_cdf_h(y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    generalised inverse of cdf of h

    :param y: 1d numpy array of input values in the range of [0, 1]
    :param alpha: value of parameter alpha
    :param beta: value of parameter beta
    :return: 1d numpy array of function values
    """
    pass

def generate_samples(sample_size: int, param_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    function which generates samples from the distributions w.r.t. the pdfs f, g, h. these samples are returned
    to the caller.

    :param sample_size: size of sample to be generated
    :param param_dict: dictionary with keys 'a', 'b', 'lam', 'alpha', 'beta' containing the parameters of
        the densities f, g, h
    :return: tuple of three 1d numpy arrays containing the samples generated from the distributions w.r.t. to f, g, h
    """
    pass

def visualise(sample_f: np.ndarray, sample_g: np.ndarray, sample_h: np.ndarray, param_dict: Dict[str, Any]):
    """
    function for the visualisation of the generated samples. for each sample a histogram depicting relative
    frequencies and the corresponding density function are generated.

    :param sample_f: 1d numpy array of samples from distribution with density f
    :param sample_g: 1d numpy array of samples from distribution with density g
    :param sample_h: 1d numpy array of samples from distribution with density h
    :param param_dict: dictionary with keys 'a', 'b', 'lam', 'alpha', 'beta' containing the parameters of
        the densities f, g, h
    :return: /
    """
    pass

def main():
    param_dict = {
        'a': -1,
        'b': 1,
        'lam': 3,
        'alpha': 3,
        'beta': 5}

    sample_size = 'fill me'
    sample_f, sample_g, sample_h = generate_samples(sample_size, param_dict)
    visualise(sample_f, sample_g, sample_h, param_dict)

if __name__ == '__main__':
    main()