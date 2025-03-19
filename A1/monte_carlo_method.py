import numpy as np
from typing import Tuple

from box_muller_method import marsaglia_bray_scheme, box_muller_scheme, affine_linear_transformation

def f(x1: np.array, x2: np.array) -> np.ndarray:
    """
    function describing the transformation of the normal random vector (X1, X2)

    :param x1: 1d numpy array
    :param x2: 1d numpy array
    :return: 1d numpy array
    """
    pass

def g(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def h(x: np.ndarray) -> np.ndarray:
    mask1 = np.logical_and(0 <= x, x < 0.5)
    mask2 = np.logical_and(0.5 <= x, x <= 1.0)

    ret_val = np.zeros_like(x)
    ret_val[mask1] = 0.5 * x[mask1]
    ret_val[mask2] = 0.5 * (1 - x[mask2])
    return ret_val

def monte_carlo(normal_sample: np.ndarray) -> Tuple[float, float]:
    """
    this function applies the monte-carlo method to compute approximations to
    expected value and variance

    :param normal_sample: sample of normal distribution
    :return: tuple containing the estimation of expected value and variance
    """
    pass

def monte_carlo_variance_reduction(sample_size: int):
    """
    this function computes the exact expected value of g(U), its vanilla monte-carlo estimate and
    the variance reduction monte-carlo estimate w.r.t. the control variate h(U).

    :param sample_size:
    :return: tuple consisting of exact expected value, vanilla monte-carlo estimate and variance
        reduction monte-carlo estimate.
    """
    pass

def main():
    np.random.seed(123)

    # vanilla monte-carlo method for the estimation of expected value and variance
    sample_size = 'fill me'

    b = 'fill me'
    a = 'fill me'

    standard_normal_sample_1 = box_muller_scheme(sample_size)
    normal_sample_1 = affine_linear_transformation(standard_normal_sample_1, a, b)
    expected_value_1, variance_1 = monte_carlo(normal_sample_1)

    standard_normal_sample_2 = marsaglia_bray_scheme(sample_size)
    normal_sample_2 = affine_linear_transformation(standard_normal_sample_2, a, b)
    expected_value_2, variance_2 = monte_carlo(normal_sample_2)

    print('expected values:')
    print(' > box muller: {:.7f}'.format(expected_value_1))
    print(' > marsaglia bray: {:.7f}'.format(expected_value_2))

    print('variances:')
    print(' > box muller: {:.7f}'.format(variance_1))
    print(' > marsaglia bray: {:.7f}'.format(variance_2))

    # monte-carlo method with variance reduction
    exact_val, mc_vanilla, mc_variance_reduction = monte_carlo_variance_reduction(sample_size)
    print('expected value')
    print(' > exact: {:.7f}'.format(exact_val))
    print(' > vanilla monte-carlo: {:.7f}'.format(mc_vanilla))
    print(' > control variate monte-carlo: {:.7f}'.format(mc_variance_reduction))

if __name__ == '__main__':
    main()