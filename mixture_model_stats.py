# Standard Python
import math

# Common packages
import torch

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
EPS_FOR_LOG = 1e-8
EPS_FOR_SINGULARITY = 1e-4

# TODO: Write a test for this


def univariate_log_likelihood(x, log_p, mu, sigma_inv):
    """
    Inputs:
       x: tensor of shape tensor(mb_size,1) containing the observed values

       log_p: tensor of shape (mb_size, mixture_componente) containing the log
       probability of each component.

       mu: tensor of shape (mb_size, mixture_components, 1) containing the mu
       estimate for each component

       sigma_inv: tensor of shape (mb_size, mixture_components, 1, 1) containing the
       estimate of the reciprocal of the sqrt of the variance for each component

    Output:
       tensor of shape (mb_size,) containing the log likelihood for each sample
       in the batch

    Note: The dimensions of the input tensors have been chosen for compatability
    with a multivarate version of this function.  The dimensions associated with
    the number of symbols are required to be 1.

    """
    # Verify that the dimensions associated with the number of symbols are all 1.
    if (
        x.shape[1] != 1
        or mu.shape[2] != 1
        or sigma_inv.shape[2] != 1
        or sigma_inv.shape[3] != 1
    ):
        raise ValueError(
            "Univariate code requires dimensions corresponding to number of symbols to be one."
        )

    # Drop the dimensions associated with the number of symbols.
    x = x.squeeze(1)
    mu = mu.squeeze(2)
    sigma_inv = sigma_inv.squeeze(3).squeeze(2)

    mb_size = x.shape[0]
    if x.shape[0] != mb_size or mu.shape != log_p.shape or mu.shape != sigma_inv.shape:
        raise ValueError("Dimensions of x, log_p, mu, and sigma_inv are inconsistent")

    # Subtract mu from x in each component.
    # Be explicit rather than relying on broadcasting
    e = x.unsqueeze(1).expand(mu.shape) - mu

    z_squared = (sigma_inv * e) ** 2

    # Inclusion of EPS is to ensure argument remains bounded away from zero.
    log_sigma_inv = torch.log(EPS_FOR_LOG + torch.abs(sigma_inv))

    # log_p, z_squared, and log_sigma_inv have the same shape: (mb_size, mixture_components)

    ll_components = log_p - 0.5 * z_squared + log_sigma_inv - LOG_SQRT_TWO_PI

    # Now sum over the mixture components with logsumexp to get the liklihoods
    # for each batch sample
    ll = torch.logsumexp(ll_components, dim=1)
    return ll


def multivariate_log_likelihood(x, log_p, mu, sigma_inv):
    """Inputs:
       x (tensor(mb_size, channels)): values
       log_p (tensor(mb_size, mixture_componente)):
                      log probability of each component (this code assumes
                      these have been normalized with logsumexp!!)
       mu (tensor(mb_size, mixture_components, channels): mu for each component
       sigma_inv (tensor(mb_size, mixture_components, channels, channels)):
                     - sqrt of inverse of covariance matrix
                       (More specifically, the inverse of the lower triangular
                       Cholesky factor of the channel covariances so that
                       C^{-1} = L^T L)


    Output:
       tensor(mb_size): log likelihood for each sample in batch

    """
    mb_size, channels = x.shape
    mixture_components = mu.shape[1]

    # Ensure the sigma_inv matrix is lower triangular
    # Values in the upper triangle are ignored
    sigma_inv = torch.tril(sigma_inv)

    e = x.unsqueeze(1).expand(mb_size, mixture_components, channels) - mu
    # e is (mb_size, mixture_components, channels)

    e = e.unsqueeze(3)
    # e is now (mb_size, mixture_components, channels, 1)

    z = torch.matmul(sigma_inv, e)
    # z is (mb_size, mixture_components, channels, 1)

    z_squared = torch.sum((z ** 2).squeeze(3), dim=2)
    # z_squared is (mb_size, mixture_components)

    # print('x: ', x)
    # print('mu: ', mu)
    # print('e: ', e)
    # print('z_squared: ', z_squared)

    # Compute the log of the diagonal entries of the inverse covariance matrix
    # Inclusion of EPS is to ensure argument stays well above zero.
    log_diag_sigma_inv = torch.log(
        EPS_FOR_LOG + torch.abs(torch.diagonal(sigma_inv, 0, -2, -1))
    )
    # log_diag_sigma_inv is (mb_size, mixture_components, channels)

    # Compute the log of the determinant of the inverse covariance
    # matrix by summing the above
    log_det_sigma_inv = torch.sum(log_diag_sigma_inv, dim=2)
    # print('log_det_sigma_inv', log_det_sigma_inv)
    # log_det_sigma_inv is (mb_size, mixture_components)

    ll_components = (
        log_p - 0.5 * z_squared + log_det_sigma_inv - channels * LOG_SQRT_TWO_PI
    )

    # Now sum over the components with logsumexp to get the liklihoods
    # for each batch sample
    ll = torch.logsumexp(ll_components, dim=1)
    return ll


def univariate_combine_metrics(p, mu, sigma_inv):
    """
    Given a mixture model of normal distributions charaterized by probabilities
    (p), components-wise mean (mu) and component-wise inverse standard deviation
    (sigma_inv), compute the overall mean and inverse standard deviation for the
    mixture.

    Note:  This assumes a univariate mu and sigma_inv.

    Inputs:
        p (tensor(mb_size, mixture_componente)):
           probability of each component
        mu (tensor(mb_size, mixture_components, symbols): mu for each component
        sigma_inv (tensor(mb_size, mixture_components, symbols, symbols): sigma for each component

    Outputs:
        return (tensor(mb_size)): expected return
        std_dev (tensor(mb_size, symbols, symbols)): std deviations (sqrt of covariance matrix)
    """

    mb_size, mixture_components, symbols = mu.shape
    if symbols != 1:
        raise ValueError("This code requires the number of symbols to be 1")

    # Drop symbols dimension on mu and sigma_inv which is known to be 1
    sigma_inv = sigma_inv.squeeze(3).squeeze(2)
    mu = mu.squeeze(2)

    variance = (1.0 / sigma_inv) ** 2
    composite_mean = torch.sum(p * mu, dim=1)
    print(f"composite_mean: {composite_mean}")

    # TODO: Verify this is correct implementation of parallel axis theorem
    # E[(x-mu)**2] = sum p_i E[(x_i-mu)**2]
    # E[(x_i-mu)**2] = E[((x_i-mu_i) + (mu_i-mu))**2]
    #                = sigma_i**2 + (mu_i-mu)**2
    shifted_component_variances = variance + (mu - composite_mean.unsqueeze(1)) ** 2
    composite_std_dev = torch.sqrt(torch.sum(p * shifted_component_variances, dim=1))
    return composite_mean, composite_std_dev
