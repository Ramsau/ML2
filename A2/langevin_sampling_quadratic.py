from typing import Tuple, List, Callable, Optional
import numpy as np
from matplotlib import pyplot as plt

def energy_func_quadratic(x: np.ndarray, Q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    this function computes the energy function of the quadratic model corresponding to the matrix Q and the
    offset mu

    :param x: 2d numpy array of shape (2, batch_size)
    :param Q: 2d numpy array
    :param mu: 2d numpy array of shape (2, 1)
    :return: numpy array
    """
    return 0.5 * np.sum((x - mu) * np.matmul(Q, x - mu), axis=0)

def grad_energy_func_quadratic(x: np.ndarray, Q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    this function computes the gradient of the quadratic model with matrix Q and offset mu

    :param x: 2d numpy array of shape (2, batch_size)
    :param Q: 2d numpy array
    :param mu: 2d numpy array of shape (2, 1)
    :return: numpy array
    """
    return np.matmul(Q, x - mu)

def density_func_quadratic(x: np.ndarray, Q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    this function computes the density (normalising constant not included) of Gibbs-type
    w.r.t. the energy function defined by energy_func_quadratic()

    :param x: 2d numpy array of shape (2, batch_size)
    :param Q: 2d numpy array
    :param mu: 2d numpy array of shape (2, 1)
    :return: numpy array
    """
    return np.exp(-energy_func_quadratic(x, Q, mu))

def expected_value_ula_quadratic(Q: np.ndarray, mu: np.ndarray, gamma: float, k: int, x0: np.ndarray) -> np.ndarray:
    """
    this function computes the exact expected value of the ULA iterate w.r.t. the quadratic
    density with matrix Q and offset mu

    :param Q: 2d numpy array
    :param mu: 2d numpy array of shape (2, 1)
    :param gamma: ULA step size
    :param k: iteration number
    :param x0: initial guess
    :return: numpy array
    """
    I = np.eye(2)
    return mu +(np.linalg.matrix_power(I- Q*gamma, k)) @ (x0 - mu)

    

def covariance_matrix_ula_quadratic(Q: np.ndarray, gamma: float, k: int) -> np.ndarray:
    """
    this function computes the exact variance-covariance matrix of the ULA iterate w.r.t. the quadratic
    density with matrix Q and offset mu

    :param Q: 2d numpy array
    :param gamma: ULA step size
    :param k: iteration number
    :return: numpy array
    """
    I = np.eye(2)
    first_term  = np.linalg.inv((Q - (gamma/2)* np.linalg.matrix_power(Q,2)))
    second_term = np.linalg.matrix_power(I - (gamma/2) * Q, k) @ first_term
    return first_term - second_term

def kullback_leibler_normals(mu_1: np.ndarray, sig_1: np.ndarray, mu_2: np.ndarray, sig_2: np.ndarray) -> np.ndarray:
    """
    this function computes the kullback-leibler divergence between the normals N_2(mu_1, sig_1), N_2(mu_2, sig_2)

    :param mu_1: expected value
    :param sig_1: variance-covariance matrix
    :param mu_2: expected value
    :param sig_2: variance-covariance matrix
    :return: numpy array of shape (num_chains, )
    """
    d = 2
    det_1 = np.linalg.det(sig_1)
    det_2 = np.linalg.det(sig_2)

    first_term = np.log(det_2/det_1)
    d = 2
    second_term = (mu_1 - mu_2).T @ np.linalg.inv(sig_2) @ (mu_1 - mu_2)
    third_term = np.trace(np.linalg.inv(sig_2) @ sig_1)
    return 0.5 * (first_term + second_term + third_term - d)

def ula_quadratic(Q: np.ndarray, mu: np.ndarray, num_iterations: int,
                  gamma: float, x0: np.ndarray) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    this function implements the ULA scheme to generate samples from the Gibbs-type distribution with quadratic
    energy function corresponding to the matrix Q and the offset mu.

    NOTE
        > the initial value x0 is assumed to be of shape (2, num_chains) - if num_chains > 1, the algorithm creates
            num_chains independent chains of ULA iterates.

    :param Q: 2d numpy array
    :param mu: 2d numpy array of shape (2, 1)
    :param num_iterations: number of ULA steps
    :param gamma: ULA step size
    :param x0: initial value of shape (2, num_chains)
    :return: numpy array of shape (num_iterations, 2, num_chains) of the ULA-iterates, list of kl-divergences
        (averaged over chains) between ULA-distribution and target distribution, list of kl-divergences between
        ULA-distribution and limit distribution,
    """
    num_chains = x0.shape[1]
    langevin_sample_ula = np.zeros((num_iterations, 2, num_chains))
    kl_div_sig = []
    kl_div_sig_infty = []
    target_mu = mu
    target_sig = np.linalg.inv(Q)
    limit_mu = mu
    limit_sig = np.linalg.inv(Q-0.5 * gamma * np.linalg.matrix_power(Q, 2))

    x_curr = x0.copy()
    for k in range(num_iterations):
        langevin_sample_ula[k, :, :] = x_curr
        x_curr = x_curr - gamma * grad_energy_func_quadratic(x_curr, Q, mu) + np.sqrt(2 * gamma) * np.random.randn(2, num_chains)
       
        mu_k = expected_value_ula_quadratic(Q, mu, gamma, k+1, x0)
        sig_k = covariance_matrix_ula_quadratic(Q, gamma, k+1)
        kl_div_sig.append(np.mean(kullback_leibler_normals(mu_k, sig_k, target_mu, target_sig)))
        kl_div_sig_infty.append(np.mean(kullback_leibler_normals(mu_k, sig_k, limit_mu, limit_sig)))


    return langevin_sample_ula, kl_div_sig, kl_div_sig_infty
    
    

    

def mala(grad_energy_func: Callable, density_func: Callable, num_iterations: int,
         gamma: float, x0: np.ndarray) -> np.ndarray:
    """
    this function implements the MALA scheme to generate samples from distribution with Gibbs-type PDF proportional
    to density_func, and energy gradient function grad_energy_func

    NOTE
        > the initial value x0 is assumed to be of shape (2, num_chains) - if num_chains > 1, the algorithm creates
            num_chains independent chains of MALA iterates.

    :param grad_energy_func: function returning to each state the gradient of the energy function
    :param density_func: function returning to each state the PDF of the target distribution
    :param num_iterations: number of  MALA steps to be performed
    :param gamma: step size of Langevin update
    :param x0: initial guess of shape (2, num_chains)
    :return: numpy array of shape (num_iterations, 2, num_chains) of the ULA-iterates
    """
    
    

    num_chains = x0.shape[1]
    langevin_sample_mala = np.zeros((num_iterations, 2, num_chains))
    x_curr = x0.copy()
    for k in range(num_iterations):
        langevin_sample_mala[k, :, :] = x_curr
        x_prev = x_curr.copy()
        x_curr = x_prev - gamma * grad_energy_func(x_prev) + np.sqrt(2 * gamma) * np.random.randn(2, num_chains)

        u = np.random.rand(num_chains)
        # compute the transition probabilities
        trans_prob_forward= np.exp(np.sum((x_curr - x_prev + gamma*grad_energy_func(x_prev))**2, axis=0)/ (4 * gamma))
        trans_prob_backward = np.exp(np.sum((x_prev - x_curr + gamma*grad_energy_func(x_curr))**2, axis=0)/(4 * gamma))

        # compute the acceptance probability
        p_accept = np.minimum(1, (trans_prob_forward * density_func(x_curr))/(trans_prob_backward * density_func(x_prev)))
        
        # accept or reject the new sample
        accept_mask = u <= p_accept

        # apply the acceptance mask
        x_curr = np.where(accept_mask, x_curr, x_prev)





        
        
    return langevin_sample_mala
        
def visualise_kl_divs(kl_div_sig: List[float], kl_div_sig_infty: List[float]):
    """
    this function creates a log-log plot depicting the evolution of the kl-divergence between the iterates
    distribution, the target and limit distribution respectively

    :param kl_div_sig: kl-divergence between iterate distribution and target distribution
    :param kl_div_sig_infty: kl-divergence between iterate distribution and limit distribution
    :return: /
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(0, len(kl_div_sig)), kl_div_sig, label='target distribution', color='cyan')
    ax.plot(np.arange(0, len(kl_div_sig)), kl_div_sig_infty, label='limit distribution', color='magenta')
    ax.set_xlabel('iteration')
    ax.set_ylabel('kl-div')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('kl divergence w.r.t. to target distribution (log-log)')
    ax.legend()
    plt.tight_layout()

def visualise_density_quadratic(Q: np.ndarray, Q_infty: np.ndarray, mu: np.ndarray):
    """
    this function generates plots visualising the target and the limit distribution generated by the ULA
    scheme for quadratic energy with matrix Q and offset mu

    :param Q: 2d numpy array
    :param Q_infty: 2d numpy array
    :param mu: 2d numpy array of shape (2, 1)
    :return: /
    """

    a = 10
    x = np.linspace(-a, a, 100)
    y = np.linspace(-a, a, 100)
    xx, yy = np.meshgrid(x, y)
    uu = np.vstack([xx.flatten(), yy.flatten()])
    zz_target = density_func_quadratic(uu, Q, mu)
    zz_infty = density_func_quadratic(uu, Q_infty, mu)

    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.pcolormesh(xx, yy, zz_target.reshape(xx.shape), cmap='viridis')
    ax_1.set_aspect('equal')
    ax_1.set_xlim([-a, a])
    ax_1.set_ylim([-a, a])
    ax_1.set_title('target distribution')

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.pcolormesh(xx, yy, zz_infty.reshape(xx.shape), cmap='viridis')
    ax_2.set_aspect('equal')
    ax_2.set_xlim([-a, a])
    ax_2.set_ylim([-a, a])
    ax_2.set_title('limit distribution')
    plt.tight_layout()

def density_func_normal(x: np.ndarray, sig_sq: float, mu: float) -> np.ndarray:
    """
    implementation of the PDF of the normal distribution

    :param x: 1d numpy array
    :param sig_sq: variance
    :param mu: expected value
    :return: numpy array
    """
    return 1 / np.sqrt(2 * np.pi * sig_sq) * np.exp((-0.5 / sig_sq) * (x - mu) ** 2)

def visualise_sample_quadratic(sample: np.ndarray, Q: np.ndarray, Q_infty: Optional[np.ndarray],
                               mu: np.ndarray):
    """
    this function generates a 2d histogram of the samples generated by one of the schemes for sampling from
    the Gibbs-distribution with quadratic density corresponding to Q, mu. adjacent to the 2d histogram
    the marginal densities and the marginal histograms are depicted

    NOTE:
        > if Q_infty is None, the marginal density w.r.t. the limit distribution is not visualised

    :param sample: 2d numpy array of shape (num_samples, 2)
    :param Q: 2d numpy array representing the matrix of the quadratic energy w.r.t. the target distribution
    :param Q_infty: 2d numpy array representing the matrix of the quadratic energy w.r.t. the limit distribution
    :param mu: 2d numpy array
    :return: /
    """
    a = 10
    sig = np.linalg.inv(Q)

    sig_infty = None
    if Q_infty is not None:
        sig_infty = np.linalg.inv(Q_infty)

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 4)

    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_main.hist2d(sample[:, 0], sample[:, 1], cmap='viridis', rasterized=False,
                   bins=128, range=np.array([(-10, 10), (-10, 10)]))
    ax_main.set_aspect('equal')
    ax_main.set_xlim(sample[:, 0].min(), sample[:, 0].max().max())
    ax_main.set_ylim(sample[:, 1].min(), sample[:, 1].max().max())

    t = np.linspace(-a, a, 100)
    marginal_target_density_x = density_func_normal(t, sig[0, 0], mu[0])
    marginal_target_density_y = density_func_normal(t, sig[1, 1], mu[1])
    if sig_infty is not None:
        marginal_limit_density_x = density_func_normal(t, sig_infty[0, 0], mu[0])
        marginal_limit_density_y = density_func_normal(t, sig_infty[1, 1], mu[1])

    ax_marginal_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_marginal_x.hist(sample[:, 0], density=True)
    ax_marginal_x.plot(t, marginal_target_density_x, color='orange', label='marginal_target')
    ax_marginal_x.set_xlim([-a, a])
    ax_marginal_x.set_yticks([])

    ax_marginal_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_marginal_y.hist(sample[:, 1], orientation='horizontal', density=True)
    ax_marginal_y.plot(marginal_target_density_y, t, color='orange', label='marginal_target')
    ax_marginal_y.set_ylim([-a, a])

    if sig_infty is not None:
        ax_marginal_x.plot(t, marginal_limit_density_x, color='cyan', label='marginal_limit')
        ax_marginal_y.plot(marginal_limit_density_y, t, color='cyan', label='marginal_limit')

    ax_marginal_x.legend()
    ax_marginal_y.legend()
    plt.tight_layout()
def main():
    mu = np.zeros(2).reshape(-1, 1)
    Q = np.array([[4, 1], [1, 2]])

    gamma = 0.4
    Q_infty = Q - 0.5 * gamma * np.linalg.matrix_power(Q, 2)

    num_iterations = 5000
    num_chains = 10

    x0 = 2 * (np.random.rand(2, num_chains) - 1)

    # NOTE
    #   > initial iterates may step from limit distribution only
    #       small probilities. this is way one considers only iterates
    #       generated after a so-called burn-in period.
    #   > the variable burn_in gives the index starting at which
    #       iterates from each chain will be considered for visualisation.

    burn_in = 3000

    # #############################################################
    # ### ULA #####################################################
    # #############################################################

    # langevin_sample_ula_quadratic, kl_div_list_sig, kl_div_list_sig_infty = ula_quadratic(Q, mu, num_iterations, gamma, x0)
    # langevin_sample_ula_quadratic = np.hstack(langevin_sample_ula_quadratic[burn_in::, :, :]).transpose()
    # visualise_kl_divs(kl_div_list_sig, kl_div_list_sig_infty)
    # visualise_sample_quadratic(langevin_sample_ula_quadratic, Q, Q_infty, mu)
    # plt.show()
    # plt.close('all')

    # #############################################################
    # ### MALA ####################################################
    # #############################################################

    langevin_sample_mala_quadratic = mala(lambda x: grad_energy_func_quadratic(x, Q, mu),
                                          lambda x: density_func_quadratic(x, Q, mu), num_iterations, gamma, x0)
    langevin_sample_mala_quadratic = np.hstack(langevin_sample_mala_quadratic[burn_in::, :, :]).transpose()
    visualise_sample_quadratic(langevin_sample_mala_quadratic, Q, None, mu)

    plt.show()
    plt.close('all')

if __name__ == '__main__':
    np.random.seed(123)
    main()