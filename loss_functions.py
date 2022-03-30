# Standard Python
import math

# Common packages
import torch

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
EPS_FOR_LOG = 1e-8
EPS_FOR_SINGULARITY = 1e-4

# TODO: Write a test for this


def ensure_nonsingular(x):
    """
    Ensure that a lower triangular matrix is non-singular by adding a small perturbation
    when necessary.  The goal here is not to touch entries that don't need a perturbation and
    to make the minimum change to reach the threshhold.
    """
    d = torch.diagonal(x, dim1=2, dim2=3)
    pos = ((d < EPS_FOR_SINGULARITY) * (d >= 0.0)).type(torch.float32)
    neg = ((d > -EPS_FOR_SINGULARITY) * (d < 0.0)).type(torch.float32)
    perturbation = pos * (EPS_FOR_SINGULARITY - d) + neg * (-EPS_FOR_SINGULARITY - d)
    return x + torch.diag_embed(perturbation)


def combine_mixture_metrics(p, mu, sigma):
    """
    Inputs:
        p (tensor(mb_size, mixture_componente)):
           probability of each component
        mu (tensor(mb_size, mixture_components): mu for each component
        sigma (tensor(mb_size, mixture_components): sigma for each component

    Outputs:
        gain (tensor(mb_size)): expected gain (in log-normal space)
        std_dev (tensor(mb_size)): std deviations (in log-normal sape)
    """
    sigma_squared = sigma ** 2
    component_means = torch.exp(mu + 0.5 * sigma_squared)
    component_variances = (torch.exp(sigma_squared) - 1.0) * component_means ** 2
    mean = torch.sum(p * component_means, dim=1)
    # TODO: Verify this is correct implementation of parallel axis theorem
    shifted_component_variances = (
        component_variances + (mean.unsqueeze(1) - component_means) ** 2
    )
    std_dev = torch.sqrt(torch.sum(p * shifted_component_variances, dim=1))
    return mean - 1.0, std_dev


def multivariate_stats(x):
    """Inputs:
      x (tensor(mb_size, channels)): values
    Outputs:
      mu (tensor(channels)): channel means
      sigma (tensor(channels, channels)): cholesky factor of cov matrix
    """
    mb_size, channels = x.shape
    mu = torch.mean(x, dim=0)
    error = x - mu.unsqueeze(0).expand((mb_size, channels))
    # error is mb_size x channels
    error1 = error.unsqueeze(2)
    # error1 represents e (mb_size, channels, 1)
    error2 = error.unsqueeze(1)
    # error2 represents e^T (mb_size, 1, channels)
    cov = torch.mean(torch.matmul(error1, error2), dim=0)
    # cov is (channels, channels)

    # Return cholesky factor
    sigma = torch.cholesky(cov)
    return mu, sigma


def univariate_mixture_log_likelihood(x, log_p, mu, sigma_inv):
    """
    Inputs:
       x: tensor of shape tensor(mb_size,) containing the observed values

       log_p: tensor of shape (mb_size, mixture_componente) containing the log
       probability of each component.

       mu: tensor of shape (mb_size, mixture_components) containing the mu
       estimate for each component

       sigma_inv: tensor of shape (mb_size, mixture_components) containing the
       estimate of the reciprocal of the sqrt of the variance for each component

    Output:
       tensor of shape (mb_size,) containing the log likelihood for each sample
       in the batch
    """
    mb_size, mixture_components = log_p.shape
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


def multivariate_mixture_log_likelihood(x, log_p, mu, sigma_inv):
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
