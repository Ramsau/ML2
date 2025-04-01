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
    return np.where(
        (a <= x) & (x <= b),
        1 / (b - a),
        0
    )

def pdf_g(x: np.ndarray, lam: float) -> np.ndarray:
    """
    implementation of pdf g

    :param x: 1d numpy array of input values in the range of (-inf, inf)
    :param lam: value of parameter lambda
    :return: 1d numpy array of function values
    """
    return np.where(
        x >= 0,
        lam * np.exp(-lam * x),
        0
    )

def pdf_h(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    implementation of pdf h

    :param x: 1d numpy array of input values in the range of (-inf, inf)
    :param alpha: value of parameter alpha
    :param beta: value of parameter beta
    :return: 1d numpy array of function values
    """
    return np.where(
        (alpha <= x) & (x <= beta),
        1 / (x * np.log(beta / alpha)),
        0
    )

def inverse_cdf_f(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    generalised inverse of cdf of f

    :param y: 1d numpy array of input values in the range of [0, 1]
    :param a: value of parameter a
    :param b: value of parameter b
    :return: 1d numpy array of function values
    """
    return y * (b - a) + a

def inverse_cdf_g(y: np.ndarray, lam: float) -> np.ndarray:
    """
    generalised inverse of cdf of g

    :param y: 1d numpy array of input values in the range of [0, 1]
    :param lam: value of parameter lambda
    :return: 1d numpy array of function values
    """
    return np.log(1 - y) / -lam

def inverse_cdf_h(y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    generalised inverse of cdf of h

    :param y: 1d numpy array of input values in the range of [0, 1]
    :param alpha: value of parameter alpha
    :param beta: value of parameter beta
    :return: 1d numpy array of function values
    """
    return alpha * np.exp(
        y * np.log(beta / alpha)
    )

def generate_samples(sample_size: int, param_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    function which generates samples from the distributions w.r.t. the pdfs f, g, h. these samples are returned
    to the caller.

    :param sample_size: size of sample to be generated
    :param param_dict: dictionary with keys 'a', 'b', 'lam', 'alpha', 'beta' containing the parameters of
        the densities f, g, h
    :return: tuple of three 1d numpy arrays containing the samples generated from the distributions w.r.t. to f, g, h
    """
    u = np.random.uniform(0, 1, size=sample_size)
    y_f = inverse_cdf_f(u, a=param_dict['a'], b=param_dict['b'])
    y_g = inverse_cdf_g(u, lam=param_dict['lam'])
    y_h = inverse_cdf_h(u, alpha=param_dict['alpha'], beta=param_dict['beta'])
    return y_f, y_g, y_h


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
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    distributions = [(sample_f, 'Distribution F', pdf_f, (param_dict['a'], param_dict['b'])),
                     (sample_g, 'Distribution G', pdf_g, (param_dict['lam'], )),
                     (sample_h, 'Distribution H', pdf_h, (param_dict['alpha'], param_dict['beta'])),]

    for ax, (samples, title, dist, params) in zip(axes, distributions):
        ax.hist(samples, bins=30, density=True, alpha=0.6, color='b', label='Histogram')

        span = max(samples) - min(samples)
        x = np.linspace(min(samples) - span * 0.1, max(samples) + span * 0.1, 1000)
        ax.plot(x, dist(x, *params), 'r-', label='Density Function')

        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    param_dict = {
        'a': -1,
        'b': 1,
        'lam': 3,
        'alpha': 3,
        'beta': 5}

    sample_size = int(1e5)
    sample_f, sample_g, sample_h = generate_samples(sample_size, param_dict)
    visualise(sample_f, sample_g, sample_h, param_dict)

if __name__ == '__main__':
    main()